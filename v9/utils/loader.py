# v6/utils/loader.py
"""v6 权重装载：Qwen3-Omni thinker（MoE）的 checkpoint 名 → 模型名映射 + 对账。

与 dense loader 的差异：
  1. checkpoint 全部带 `thinker.model.` 前缀（vision/audio tower 的权重直接跳过）；
  2. 128 个专家的 gate_proj/up_proj/down_proj 融合成 w13[E, 2I, H] / w2[E, H, I]；
  3. lm_head 是独立的（untied），名字带 `thinker.`；
  4. 模型配置从 config.json 的 thinker_config.text_config 读（嵌套）。"""
import json
import os
from glob import glob

import torch
import torch.nn as nn
from safetensors import safe_open


def load_thinker_config(model_dir: str) -> dict:
    """config.json 的 thinker_config.text_config → 平铺的模型配置。"""
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    return raw["thinker_config"]["text_config"]


def _ckpt_name(name: str):
    """checkpoint 名 → (模型名, shard)。shard=None 直装；专家的 gate/up/down 带切片信息。
    返回 (None, None) 表示跳过（vision/audio tower 等）。"""
    # thinker.lm_head → lm_head（在 thinker 下但不在 thinker.model 下）
    if name == "thinker.lm_head.weight":
        return "lm_head.weight", None
    # 多模态:vision tower(VLLM_LEARN_MM=1 时模型挂载 model.vision_tower)
    if name.startswith("thinker.visual."):
        n = "model.vision_tower." + name[len("thinker.visual."):]
        n = n.replace("patch_embed.proj.", "patch_embed.") \
             .replace("merger_list.", "mergers_ds.")
        return n, None
    if not name.startswith("thinker.model."):
        return None, None
    # thinker.model.X → model.X（我们的 Qwen3MoeModel 挂在 .model 下）
    n = "model." + name[len("thinker.model."):]
    # MoE 专家：...mlp.experts.{i}.{proj}.weight → 融合进 w13/w2 的第 i 个专家
    if ".mlp.experts." in n:
        parts = n.split(".")
        i = int(parts[parts.index("experts") + 1])
        proj = parts[-2]                      # gate_proj / up_proj / down_proj
        base = n[:n.index(".mlp.experts.")]
        if proj in ("gate_proj", "up_proj"):
            half = 0 if proj == "gate_proj" else 1
            return f"{base}.mlp.experts.w13", (i, half)
        return f"{base}.mlp.experts.w2", (i, None)
    # router：mlp.gate.weight → mlp.router.weight
    if n.endswith(".mlp.gate.weight"):
        return n.replace(".mlp.gate.weight", ".mlp.router.weight"), None
    return n, None


def _load_into(param: torch.Tensor, loaded: torch.Tensor, shard, owner):
    """weight_loader 协议：装载逻辑归组件，本函数只负责调用。
    并行组件（v9.3 ColumnParallel/RowParallel/GroupedExperts）的 weight_loader
    吃【全量】checkpoint 张量、内部切本 rank 片——所以 shape 不匹配不是错误，
    是并行的常态。优先级：param 上的 weight_loader > owner 模块的 > 形状直拷
    （直拷仅限单卡无协议组件）。"""
    wl = getattr(param, "weight_loader", None)
    if wl is None:
        if owner is not None and hasattr(owner, "weight_loader"):
            wl = owner.weight_loader
    if wl is not None:
        wl(param, loaded, shard)
        return
    assert param.shape == loaded.shape, (param.shape, loaded.shape)
    param.data.copy_(loaded)


def _load_from_slice(param, tensor_slice, shard, owner):
    """按 rank 需要的窗口从 safetensors slice 直接拷贝。

    旧路径对每个 rank 调 ``get_tensor``，即使最后只保留一小段 TP/EP
    权重，也会把完整 30B 张量读入 CPU 并产生峰值。``safe_open.get_slice``
    返回惰性切片；这里把切分规则下沉到已有组件的 weight_loader 语义，
    只 materialize 本 rank 的行/列/专家窗口。非并行参数仍走全量切片。
    """
    from layers.linear import ColumnParallelLinear, RowParallelLinear
    from layers.moe import GroupedExperts
    if isinstance(owner, ColumnParallelLinear):
        if owner.gate_up and shard in (0, 1):
            s, l = owner.shard_start, owner.shard_len
            v = tensor_slice[s:s + l, :]
            param.data[shard * l:(shard + 1) * l].copy_(v.to(param.dtype))
        elif owner.gate_up:
            # Legacy checkpoints may store an already fused gate|up tensor.
            _load_into(param, tensor_slice[:], shard, owner)
        else:
            v = tensor_slice[owner.shard_start:owner.shard_start + owner.shard_len, :]
            param.data.copy_(v.to(param.dtype))
        return
    if isinstance(owner, RowParallelLinear):
        v = tensor_slice[:, owner.shard_start:owner.shard_start + owner.shard_len]
        param.data.copy_(v.to(param.dtype)); return
    if isinstance(owner, GroupedExperts):
        i, half = shard
        if not (owner.e_start <= i < owner.e_start + owner.num_experts):
            return
        li = i - owner.e_start
        s, e = owner.i_start, owner.i_start + owner.inter
        if half is None:
            v = tensor_slice[:, s:e]
            param.data[li].copy_(v.to(param.dtype))
        else:
            v = tensor_slice[s:e, :]
            param.data[li, half * owner.inter:(half + 1) * owner.inter].copy_(v.to(param.dtype))
        return
    _load_into(param, tensor_slice[:], shard, owner)


def load_model(model: nn.Module, model_dir: str, device="cuda", dtype=torch.bfloat16):
    """装载 + 对账。MoE 专家按需融合（gate/up 拼 w13）。
    装载逻辑走 weight_loader 协议（组件自持），本函数只做名字映射和调用。"""
    files = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    assert files, f"没找到 safetensors: {model_dir}"

    # param → 所属模块 的查找表（weight_loader 协议的调用目标）
    param_owner = {id(p): mod for mod in model.modules()
                   for p in mod.parameters(recurse=False)}

    loaded = set()
    skipped = set()
    def _target_exists(target):
        try:
            model.get_parameter(target)
            return True
        except AttributeError:
            return False

    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                target, shard = _ckpt_name(name)
                if target is None:
                    skipped.add(name.split(".")[1] if name.startswith("thinker.") else name)
                    continue
                if not _target_exists(target):
                    skipped.add(target)
                    continue
                param = model.get_parameter(target)
                owner = param_owner.get(id(param))
                # EP expert shards which belong to another rank are skipped
                # before touching storage; TP/EP local windows use get_slice.
                if owner is not None:
                    try:
                        from layers.moe import GroupedExperts
                        if isinstance(owner, GroupedExperts) and shard is not None:
                            i, _ = shard
                            if not (owner.e_start <= i < owner.e_start + owner.num_experts):
                                continue
                    except ImportError:
                        pass
                if hasattr(f, "get_slice"):
                    _load_from_slice(param, f.get_slice(name), shard, owner)
                else:
                    _load_into(param, f.get_tensor(name), shard, owner)
                loaded.add(target)

    model_keys = set(model.state_dict().keys())
    not_loaded = model_keys - loaded
    # 非本 stage 的 decoder 层使用 PPMissingLayer，不产生 state_dict 项。
    assert not not_loaded, f"以下参数从未被加载: {sorted(not_loaded)[:10]}"

    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f"[loader] 装载 {len(loaded)} 个参数张量（专家已融合为 w13/w2），"
          f"跳过非 thinker 权重 {len(skipped)} 组")
    return model
