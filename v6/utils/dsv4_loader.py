# v6/utils/dsv4_loader.py
"""DeepSeek-V4 的权重装载（FP8 反量化 + 名字映射 + weight_loader 协议）。

与 qwen 系 loader 的差异：
1. FP8 block 量化权重（int8 存储 + e8m0 scale）在装载时反量化为 bf16；
2. 专家是 w1/w2/w3 三矩阵（非 gate/up 融合）；
3. HC（hyper-connections）的 fn/base/scale 直装；
4. hash 层的 tid2eid 是 buffer（long 查表）。
"""
import glob
import json
import os

import torch
import torch.nn as nn
from safetensors import safe_open

BLOCK = 128


_FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequant_fp4(packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """MXFP4 路由专家：int8 双 nibble 打包 + e8m0 scale（(1, 32) 块）→ bf16。
    每字节两个 4bit e2m1（低半在前），scale 是 [rows, cols/32]。"""
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32)
    u8 = packed.contiguous().view(torch.uint8)
    low = (u8 & 0xF).long()
    high = ((u8 >> 4) & 0xF).long()
    w = torch.stack([lut[low], lut[high]], dim=-1).reshape(
        *packed.shape[:-1], 2 * packed.shape[-1])          # [H, W_logical]
    s = scale_e8m0.float()
    H, W = w.shape
    out = w.clone()
    # (1, 32) 块：每行按 32 列一段乘 scale[row, col//32]
    out = w.view(H, W // 32, 32) * s.unsqueeze(-1)
    return out.reshape(H, W).to(torch.bfloat16)


def dequant_fp8(w_stored: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """FP8 + e8m0 scale（[H/128, W/128] 块）→ bf16。
    入参两种 dtype：int8（需位 cast + NaN pad 清理）或 float8_e4m3fn（直接转）。"""
    if w_stored.dtype == torch.int8:
        w = w_stored.clone()
        w[(w == 127) | (w == -1)] = 0       # e4m3fn 的 NaN pad 字节
        w = w.view(torch.float8_e4m3fn).float()
    else:
        w = w_stored.float()
        w[w.isnan()] = 0                    # fp8 dtype 直读的 NaN pad
    s = scale_e8m0.float()                     # e8m0 的 .float() 已是 2^(bits-127)
    H, W = w.shape
    out = torch.empty(H, W, dtype=torch.float32)
    for i in range(0, H, BLOCK):
        for j in range(0, W, BLOCK):
            out[i:i + BLOCK, j:j + BLOCK] = w[i:i + BLOCK, j:j + BLOCK] * s[i // BLOCK, j // BLOCK]
    return out.to(torch.bfloat16)


def _map_name(name: str):
    """checkpoint 名 → (模型名, shard_id)。返回 (None, None) 跳过。"""
    # 直接匹配的（embed/norm/hc/lm_head 等）
    if name in ("embed.weight",):
        return "model.embed.weight", None
    if name == "head.weight":                   # checkpoint 里叫 head 不叫 lm_head
        return "lm_head.weight", None
    if name == "norm.weight":
        return "model.norm.weight", None
    if name in ("hc_head_base", "hc_head_fn", "hc_head_scale"):   # 顶层 HC 无点分隔
        return "model.hc_head.hc_" + name[len("hc_head_"):], None  # HyperHead 参数带 hc_ 前缀
    if not name.startswith("layers."):
        return None, None
    parts = name.split(".")
    li = int(parts[1])
    rest = parts[2:]
    P = f"model.layers.{li}"

    # ---- attention ----
    if rest[0] == "attn":
        sub = rest[1]
        m = {
            "q_norm": ("self_attn.q_a_norm.weight", None),
            "kv_norm": ("self_attn.kv_norm.weight", None),
            "attn_sink": ("self_attn.sinks", None),
            "wq_a": ("self_attn.q_a_proj.weight", None),
            "wq_b": ("self_attn.q_b_proj.weight", None),
            "wkv": ("self_attn.kv_proj.weight", None),
            "wo_a": ("self_attn.o_a.weight", None),
            "wo_b": ("self_attn.o_b.weight", None),
        }
        if sub == "attn_sink":                 # layers.X.attn.attn_sink（无后缀）
            return f"{P}.self_attn.sinks", None
        if sub in m and rest[-1] == "weight":
            return f"{P}.{m[sub][0]}", None
        if rest[-1] == "scale":                     # FP8 scale → 与 weight 一起反量化
            return ("__scale__", name[:-len(".scale")])   # 标记: 先缓存 scale
        return None, None

    # ---- norm / hc ----
    if rest[0] == "attn_norm":
        return f"{P}.input_layernorm.weight", None
    if rest[0] == "ffn_norm":
        return f"{P}.post_attention_layernorm.weight", None
    if rest[0] in ("hc_attn_base", "hc_attn_fn", "hc_attn_scale"):
        return f"{P}.attn_hc." + rest[0][len("hc_attn_"):], None
    if rest[0] in ("hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale"):
        return f"{P}.ffn_hc." + rest[0][len("hc_ffn_"):], None

    # ---- ffn ----
    if rest[0] == "ffn":
        if rest[1] == "gate":
            if rest[2] == "tid2eid":
                return f"{P}.mlp.tid2eid", None
            if rest[2] == "bias":
                return f"{P}.mlp.gate_bias", None
            return f"{P}.mlp.gate_weight", None
        if rest[1] == "shared_experts":
            proj = {"w1": "sw1", "w2": "sw2", "w3": "sw3"}[rest[2]]
            if rest[-1] == "scale":
                return ("__scale__", name[:-len(".scale")])
            return f"{P}.mlp.{proj}", ("shared", None)
        if rest[1] == "experts":
            ei = int(rest[2])
            proj = {"w1": "w1", "w2": "w2", "w3": "w3"}[rest[3]]
            if rest[-1] == "scale":
                return ("__scale__", name[:-len(".scale")])
            return f"{P}.mlp.{proj}", ("expert", ei)
    return None, None


def load_dsv4(model: nn.Module, model_dir: str, device="cuda",
              dtype=torch.bfloat16, layer_device_map=None) -> nn.Module:
    """layer_device_map: {layer_idx: device}——反量化后立即搬到对应卡
    （DSV4 反量化后 ~150GB bf16 单卡放不下，必须分层分放）。"""
    files = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
    assert files, f"没找到分片: {model_dir}"

    def _target_device(target):
        if layer_device_map is None:
            return device
        for k, dev in layer_device_map.items():
            if f"layers.{k}." in target:
                return dev
        return layer_device_map.get(-1, device)       # -1 = 非层参数

    # 第一遍：收集全部 scale（weight 名 → scale 张量）
    scales = {}
    for fn in files:
        with safe_open(fn, "pt", "cpu") as f:
            for name in f.keys():
                if name.endswith(".scale"):
                    scales[name[:-len(".scale")]] = f.get_tensor(name)

    # param → owner 映射（weight_loader 协议）
    owner_map = {}
    for mod_name, mod in model.named_modules():
        for p_name, p in mod.named_parameters(recurse=False):
            owner_map[id(p)] = mod
        for b_name, b in mod.named_buffers(recurse=False):
            owner_map[id(b)] = mod

    loaded = set()
    for fn in files:
        with safe_open(fn, "pt", "cpu") as f:
            for name in f.keys():
                if name.endswith(".scale"):
                    continue                        # 已在第一遍收集
                target, shard = _map_name(name)
                if target is None:
                    continue
                if target == "__scale__":
                    continue
                raw = f.get_tensor(name)
                # 反量化：有 scale 伴随的权重（路由专家 MXFP4 / 其余 FP8）
                base = name[:-len(".weight")] if name.endswith(".weight") else name
                if base in scales:
                    sc = scales[base]
                    _, W = raw.shape
                    # raw 是打包宽度: FP4 的逻辑宽 = 2W（scale 列 = 2W/32 = W/16）
                    #              FP8 的逻辑宽 = W（scale 列 = W/128）
                    if sc.shape[-1] == W // 16:
                        raw = dequant_fp4(raw, sc)        # MXFP4
                    else:
                        raw = dequant_fp8(raw, sc)        # FP8 128×128 块
                if target.endswith("tid2eid"):
                    model.get_buffer(target).copy_(raw)
                    loaded.add(target)
                    continue
                param = model.get_parameter(target)
                owner = owner_map.get(id(param))
                # CPU 上完成 copy（RAM 充裕），随后立即把该参数整体搬到目标卡。
                # 注意 copy_ 是跨设备安全的：param(CPU) ← raw(已反量化)，
                # 然后用 .data = .data.to(dev) 整体替换（无额外副本——
                # 旧 CPU 张量即刻可回收）。
                dev_here = _target_device(target)
                if shard is not None and owner is not None and hasattr(owner, "weight_loader"):
                    owner.weight_loader(param, raw, shard)
                else:
                    assert param.shape == raw.shape, (target, param.shape, raw.shape)
                    param.data.copy_(raw)
                del raw                        # raw 的 CPU 副本即刻释放
                loaded.add(target)

    missing = set(model.state_dict().keys()) - loaded
    # hc_head 的 buffer 类参数（unweighted rms 无权重）等允许缺失项
    real_missing = {m for m in missing
                    if not any(x in m for x in ["tid2eid", "gate_bias"])}
    assert not real_missing, f"未加载: {sorted(real_missing)[:8]}"

    # ---- 统一搬运（全部 CPU 装载完成后）----
    # shard 路径的大参数（w1 [256,2048,4096]）中途搬运会带着未装填的垃圾
    # 占 GPU——必须等该参数的所有 shard 装完。逐参数立即搬（非 shard 的），
    # shard 的延迟到此处统一搬。
    import gc
    if layer_device_map is None:
        model = model.to(device=device, dtype=dtype)
    else:
        # 逐参数搬运已完成（含 dtype 转换）
        _dbg = [0]
        for name, p in list(model.named_parameters()):
            dev_here = _target_device(name)
            # 带 dtype 一起搬: 模型构造的 fp32 param 直接落成 bf16
            # （否则 fp32 的 w1 是 8.6GB/个，4 层就爆——本轮 OOM 的根因）
            if p.data.device != dev_here or p.data.dtype != dtype:
                p.data = p.data.to(dev_here, dtype=dtype)
        # buffers 也搬（tid2eid 等）
        for name, b in list(model.named_buffers()):
            dev_here = _target_device(name)
            if b.device != dev_here:
                b.data = b.data.to(dev_here)
        gc.collect()
    model.eval()
    print(f"[dsv4-loader] 装载 {len(loaded)} 张量（FP8 已反量化），"
          f"跳过 {len(scales)} 个 scale 伴随张量")
    return model
