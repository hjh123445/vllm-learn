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


def _load_into(param: torch.Tensor, loaded: torch.Tensor, shard):
    if shard is None:
        assert param.shape == loaded.shape, (param.shape, loaded.shape)
        param.data.copy_(loaded)
        return
    i, half = shard
    if half is not None:                       # gate/up → w13 第 i 个专家的上/下半
        I = loaded.shape[0]
        param.data[i, half * I:(half + 1) * I, :] = loaded
    else:                                      # down → w2 第 i 个专家
        param.data[i] = loaded


def load_model(model: nn.Module, model_dir: str, device="cuda", dtype=torch.bfloat16):
    """装载 + 对账。MoE 专家按需融合（gate/up 拼 w13）。"""
    files = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    assert files, f"没找到 safetensors: {model_dir}"

    loaded = set()
    skipped = set()
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                target, shard = _ckpt_name(name)
                if target is None:
                    skipped.add(name.split(".")[1] if name.startswith("thinker.") else name)
                    continue
                param = model.get_parameter(target)
                _load_into(param, f.get_tensor(name), shard)
                loaded.add(target)

    model_keys = set(model.state_dict().keys())
    not_loaded = model_keys - loaded
    assert not not_loaded, f"以下参数从未被加载: {sorted(not_loaded)[:10]}"

    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f"[loader] 装载 {len(loaded)} 个参数张量（专家已融合为 w13/w2），"
          f"跳过非 thinker 权重 {len(skipped)} 组")
    return model
