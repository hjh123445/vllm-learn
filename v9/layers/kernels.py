"""参考数学路径与 Triton 融合算子。

RMSNorm、Q/K norm + RoPE + cache store 默认走参考实现；对应 Triton
分支尚未通过整模 cached oracle，需显式开关实验。MoE SwiGLU 的 SiLU×up
auto 仅对已测的 A800、连续 BF16、intermediate=768 输入选择：少于6144行
使用 PyTorch，达到阈值使用 Triton。其他形状/设备使用 PyTorch。可通过
VLLM_LEARN_SILU_BACKEND=triton|torch 固定后端。
融合可能减少启动与中间张量开销，但加速和数值等价均需分别验证。
cache 偏移使用 int64，避免大 KV 池地址计算溢出。
"""
import os
from functools import lru_cache
import torch
import triton
import triton.language as tl


# The fused variants have not passed the full-model cached oracle. Keep the
# validated math as the default; fast kernels remain an explicit experiment.
_REFERENCE_MATH = os.getenv("VLLM_LEARN_REFERENCE_MATH", "1") == "1"
_REFERENCE_RMS = os.getenv("VLLM_LEARN_REFERENCE_RMS", str(int(_REFERENCE_MATH))) == "1"
_REFERENCE_QK = os.getenv("VLLM_LEARN_REFERENCE_QK", str(int(_REFERENCE_MATH))) == "1"
_SILU_BACKEND = os.getenv("VLLM_LEARN_SILU_BACKEND", "auto").lower()
_SILU_AUTO_TRITON_MIN_TOKENS = 6144
if _SILU_BACKEND not in ("auto", "triton", "torch"):
    raise ValueError("SiLU backend must be auto, triton, or torch")


@lru_cache(maxsize=None)
def _silu_profiled_device(device_index):
    return 'A800' in torch.cuda.get_device_name(device_index)


def _reference_rmsnorm(x, weight, eps):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return weight * value.to(x.dtype)


# ================= 残差加法 + RMSNorm 融合 =================

@triton.jit
def _add_rmsnorm_kernel(X, RES, W, OUT, N, EPS, BLOCK: tl.constexpr):
    """out = rmsnorm(x + residual) * w；residual 原地更新为 x + residual。"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + r
    x = x.to(RES.dtype.element_ty).to(tl.float32)
    var = tl.sum(x * x, 0) / N
    y = x * tl.math.rsqrt(var + EPS)
    y = y.to(OUT.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + row * N + cols, (y * w).to(OUT.dtype.element_ty), mask=mask)
    tl.store(RES + row * N + cols, x.to(RES.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor,
                      weight: torch.Tensor, eps: float):
    """返回 (normed_out, new_residual)。residual 被原地更新。"""
    if _REFERENCE_RMS:
        # HF materializes the residual addition in model dtype, then rounds
        # the normalized value before multiplying the model-dtype weight.
        residual.add_(x)
        return _reference_rmsnorm(residual, weight, eps), residual
    T, N = x.shape
    out = torch.empty_like(x)
    _add_rmsnorm_kernel[(T,)](x, residual, weight, out, N, eps,
                               BLOCK=triton.next_power_of_2(N),
                               num_warps=8 if N >= 4096 else 4, enable_fp_fusion=False)
    return out, residual


# ================= SiLU × up 融合（SwiGLU 后半）=================

@triton.jit
def _silu_mul_kernel(X, OUT, T: tl.constexpr, M: tl.constexpr,
                     BLOCK: tl.constexpr):
    """Flattened tiled SiLU×up; bounds register use for wide expert MLPs."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < T * M
    row = offsets // M
    col = offsets % M
    gate = tl.load(X + row * (2 * M) + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * (2 * M) + M + col, mask=mask, other=0.0).to(tl.float32)
    silu = (gate / (1.0 + tl.math.exp(-gate))).to(OUT.dtype.element_ty).to(tl.float32)
    y = silu * up
    tl.store(OUT + offsets, y.to(OUT.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor, backend: str | None = None) -> torch.Tensor:
    """Compute ``silu(gate) * up`` for a ``[..., 2I]`` tensor.

    ``auto`` uses the measured A800/BF16/I=768 crossover at 6144 token rows,
    and PyTorch otherwise. Explicit ``triton`` rejects unsupported inputs
    instead of silently changing the measured backend.
    """
    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("SwiGLU input must have an even final dimension")
    selected = _SILU_BACKEND if backend is None else backend.lower()
    if selected not in ("auto", "triton", "torch"):
        raise ValueError("SiLU backend must be auto, triton, or torch")
    supported_dtype = x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    triton_supported = x.is_cuda and x.is_contiguous() and supported_dtype
    if selected == "triton" and not triton_supported:
        raise ValueError("Triton SiLU requires contiguous CUDA fp16/bf16/fp32 input")

    prefix = x.shape[:-1]
    M = x.shape[-1] // 2
    if M == 0:
        return x[..., :0]
    if x.numel() == 0:
        return x.new_empty((*prefix, M))
    T = x.numel() // (2 * M)
    use_triton = selected == "triton" or (
        selected == "auto" and triton_supported and x.dtype == torch.bfloat16
        and M == 768 and T >= _SILU_AUTO_TRITON_MIN_TOKENS
        and _silu_profiled_device(x.device.index))
    if use_triton:
        flat = x.view(T, 2 * M)
        out = torch.empty((T, M), dtype=x.dtype, device=x.device)
        block = min(1024, triton.next_power_of_2(M))
        _silu_mul_kernel[(triton.cdiv(T * M, block),)](
            flat, out, T=T, M=M, BLOCK=block, num_warps=4,
            enable_fp_fusion=False)
        return out.reshape(*prefix, M)

    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


# ================= q/k norm + rope + cache 写入 融合 =================

@triton.jit
def _qk_norm_rope_store_kernel(Q, K, V, WQ, WK, QOUT, CS, POS, SLOT, KC, VC,
                              EPS,
                              HQ: tl.constexpr, HK: tl.constexpr,
                              HD: tl.constexpr, MAXB, BT_ROW_STRIDE):
    """每个 token 一个 program：
      q [HQ, HD] → rmsnorm(per head) → rope → 写回 QOUT
      k [HK, HD] → rmsnorm → rope → 散射进 cache 槽位（slot * HK*HD）
      v [HK, HD] → 原样散射进 cache
    CS 是 cos_sin_cache 表 [max_pos, HD]（前半 cos 后半 sin）。"""
    t = tl.program_id(0)
    pos = tl.load(POS + t)
    slot = tl.load(SLOT + t).to(tl.int64)
    d2 = tl.arange(0, HD // 2)
    cos = tl.load(CS + pos * HD + d2).to(tl.float32)
    sin = tl.load(CS + pos * HD + HD // 2 + d2).to(tl.float32)

    # ---- q：norm + rope ----
    hq = tl.arange(0, HQ)[:, None]
    off_q1 = t.to(tl.int64) * HQ * HD + hq * HD + d2[None, :]
    off_q2 = off_q1 + HD // 2
    x1 = tl.load(Q + off_q1).to(tl.float32)
    x2 = tl.load(Q + off_q2).to(tl.float32)
    var = (tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / HD
    inv = tl.math.rsqrt(var + EPS)
    wq1 = tl.load(WQ + d2).to(tl.float32)
    wq2 = tl.load(WQ + HD // 2 + d2).to(tl.float32)
    x1 = ((x1 * inv[:, None]).to(Q.dtype.element_ty).to(tl.float32) * wq1[None, :]).to(Q.dtype.element_ty).to(tl.float32)
    x2 = ((x2 * inv[:, None]).to(Q.dtype.element_ty).to(tl.float32) * wq2[None, :]).to(Q.dtype.element_ty).to(tl.float32)
    qc1 = (cos[None, :] * x1).to(Q.dtype.element_ty).to(tl.float32)
    qs2 = (sin[None, :] * x2).to(Q.dtype.element_ty).to(tl.float32)
    qs1 = (sin[None, :] * x1).to(Q.dtype.element_ty).to(tl.float32)
    qc2 = (cos[None, :] * x2).to(Q.dtype.element_ty).to(tl.float32)
    tl.store(QOUT + off_q1, (qc1 - qs2).to(QOUT.dtype.element_ty))
    tl.store(QOUT + off_q2, (qs1 + qc2).to(QOUT.dtype.element_ty))

    # ---- k：norm + rope → cache；v：原样 → cache ----
    hk = tl.arange(0, HK)[:, None]
    off_k1 = t.to(tl.int64) * HK * HD + hk * HD + d2[None, :]
    off_k2 = off_k1 + HD // 2
    off_c1 = slot * HK * HD + hk * HD + d2[None, :]
    off_c2 = off_c1 + HD // 2
    k1 = tl.load(K + off_k1).to(tl.float32)
    k2 = tl.load(K + off_k2).to(tl.float32)
    var = (tl.sum(k1 * k1, 1) + tl.sum(k2 * k2, 1)) / HD
    inv = tl.math.rsqrt(var + EPS)
    wk1 = tl.load(WK + d2).to(tl.float32)
    wk2 = tl.load(WK + HD // 2 + d2).to(tl.float32)
    k1 = ((k1 * inv[:, None]).to(K.dtype.element_ty).to(tl.float32) * wk1[None, :]).to(K.dtype.element_ty).to(tl.float32)
    k2 = ((k2 * inv[:, None]).to(K.dtype.element_ty).to(tl.float32) * wk2[None, :]).to(K.dtype.element_ty).to(tl.float32)
    kc1 = (cos[None, :] * k1).to(K.dtype.element_ty).to(tl.float32)
    ks2 = (sin[None, :] * k2).to(K.dtype.element_ty).to(tl.float32)
    ks1 = (sin[None, :] * k1).to(K.dtype.element_ty).to(tl.float32)
    kc2 = (cos[None, :] * k2).to(K.dtype.element_ty).to(tl.float32)
    # slot < 0 = DCP 门控丢弃（本 rank 不持有该块）——mask 掉 store。
    # mask 需与 value 同形可广播（标量不行，构 [1, HD//2]）
    keep = (d2[None, :] >= 0) & (slot >= 0)
    tl.store(KC + off_c1, (kc1 - ks2).to(KC.dtype.element_ty),
             mask=keep)
    tl.store(KC + off_c2, (ks1 + kc2).to(KC.dtype.element_ty),
             mask=keep)
    # v：[HK, HD] 原样
    d = tl.arange(0, HD)
    off_v = t.to(tl.int64) * HK * HD + hk * HD + d[None, :]
    off_cv = slot * HK * HD + hk * HD + d[None, :]
    tl.store(VC + off_cv, tl.load(V + off_v),
             mask=(d[None, :] >= 0) & (slot >= 0))


def fused_qk_norm_rope_store(q, k, v, q_norm_w, k_norm_w, cos_sin_cache,
                            positions, slot_mapping, k_cache, v_cache, eps):
    """返回 rope 后的 q（k/v 已写进 cache）。"""
    if _REFERENCE_QK:
        cache = cos_sin_cache[positions].reshape(q.shape[0], 1, q.shape[-1])
        cos, sin = cache.chunk(2, dim=-1)
        cos = torch.cat((cos, cos), dim=-1).to(q.dtype)
        sin = torch.cat((sin, sin), dim=-1).to(q.dtype)

        def rotate(value, weight):
            value = _reference_rmsnorm(value, weight, eps)
            first, second = value.chunk(2, dim=-1)
            return value*cos + torch.cat((-second, first), dim=-1)*sin

        q_out, k_out = rotate(q, q_norm_w), rotate(k, k_norm_w)
        keep = slot_mapping >= 0
        slots = slot_mapping[keep].long()
        k_cache.view(-1, k.shape[1], k.shape[2])[slots] = k_out[keep]
        v_cache.view(-1, v.shape[1], v.shape[2])[slots] = v[keep]
        return q_out
    T, HQ, HD = q.shape
    HK = k.shape[1]
    q_out = torch.empty_like(q)
    _qk_norm_rope_store_kernel[(T,)](
        q, k, v, q_norm_w, k_norm_w, q_out, cos_sin_cache, positions, slot_mapping,
        k_cache, v_cache, eps, HQ=HQ, HK=HK, HD=HD, MAXB=0, BT_ROW_STRIDE=0, enable_fp_fusion=False)
    return q_out
