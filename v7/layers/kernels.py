# v5/layers/kernels.py
"""v4 手写 triton 融合算子。

设计原则（由 profile 数据决定，见 PERFORMANCE_NOTES §7.1）：
  - 每层每步都跑的 → 手写 triton 融合（零 guard 税）
  - 每步只跑一次的 → torch.compile（guard 税一次可忽略，见 sampler.py）

融合清单（每个 kernel 替代的 aten kernel 数）：
  fused_add_rmsnorm      7 个（add + pow/mean/rsqrt/mul×2/to）—— 残差加法一起吃掉
  silu_and_mul           4 个（chunk + silu + mul）
  fused_qk_norm_rope_store  30+ 个 —— q_norm + k_norm + rope(q,k) + cache 写入
                         融成一个 kernel，q 归一化 rope 后返回，
                         k 归一化 rope 后直接散射进 cache 槽位（v 原样写入）
注意：cache 偏移一律先转 int64（13000 block × 26 万元素/block 会溢出 int32）。"""
import torch
import triton
import triton.language as tl


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
    var = tl.sum(x * x, 0) / N
    y = x * tl.math.rsqrt(var + EPS)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + row * N + cols, (y * w).to(OUT.dtype.element_ty), mask=mask)
    tl.store(RES + row * N + cols, x.to(RES.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor,
                      weight: torch.Tensor, eps: float):
    """返回 (normed_out, new_residual)。residual 被原地更新。"""
    T, N = x.shape
    out = torch.empty_like(x)
    _add_rmsnorm_kernel[(T,)](x, residual, weight, out, N, eps,
                               BLOCK=triton.next_power_of_2(N),
                               num_warps=8 if N >= 4096 else 4)
    return out, residual


# ================= SiLU × up 融合（SwiGLU 后半）=================

@triton.jit
def _silu_mul_kernel(X, OUT, M, BLOCK: tl.constexpr):
    """x: [T, 2M]（gate | up），out = silu(gate) * up。"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < M
    gate = tl.load(X + row * 2 * M + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X + row * 2 * M + M + cols, mask=mask, other=0.0).to(tl.float32)
    y = gate / (1.0 + tl.math.exp(-gate)) * up
    tl.store(OUT + row * M + cols, y.to(OUT.dtype.element_ty), mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    T, M2 = x.shape
    M = M2 // 2
    out = torch.empty((T, M), dtype=x.dtype, device=x.device)
    _silu_mul_kernel[(T,)](x, out, M, BLOCK=triton.next_power_of_2(M))
    return out


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
    x1 = x1 * inv[:, None] * wq1[None, :]
    x2 = x2 * inv[:, None] * wq2[None, :]
    tl.store(QOUT + off_q1, (cos[None, :] * x1 - sin[None, :] * x2).to(QOUT.dtype.element_ty))
    tl.store(QOUT + off_q2, (sin[None, :] * x1 + cos[None, :] * x2).to(QOUT.dtype.element_ty))

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
    k1 = k1 * inv[:, None] * wk1[None, :]
    k2 = k2 * inv[:, None] * wk2[None, :]
    tl.store(KC + off_c1, (cos[None, :] * k1 - sin[None, :] * k2).to(KC.dtype.element_ty))
    tl.store(KC + off_c2, (sin[None, :] * k1 + cos[None, :] * k2).to(KC.dtype.element_ty))
    # v：[HK, HD] 原样
    d = tl.arange(0, HD)
    off_v = t.to(tl.int64) * HK * HD + hk * HD + d[None, :]
    off_cv = slot * HK * HD + hk * HD + d[None, :]
    tl.store(VC + off_cv, tl.load(V + off_v))


def fused_qk_norm_rope_store(q, k, v, q_norm_w, k_norm_w, cos_sin_cache,
                            positions, slot_mapping, k_cache, v_cache, eps):
    """返回 rope 后的 q（k/v 已写进 cache）。"""
    T, HQ, HD = q.shape
    HK = k.shape[1]
    q_out = torch.empty_like(q)
    _qk_norm_rope_store_kernel[(T,)](
        q, k, v, q_norm_w, k_norm_w, q_out, cos_sin_cache, positions, slot_mapping,
        k_cache, v_cache, eps, HQ=HQ, HK=HK, HD=HD, MAXB=0, BT_ROW_STRIDE=0)
    return q_out
