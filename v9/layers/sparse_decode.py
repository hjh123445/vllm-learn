"""Graph-safe, replicated-input sparse EP expert core (DP=1).

Capacity is T rows per local expert, never a token-dropping limit. The GPU
compacts selected tokens; fixed launch grids read device counts and skip empty
tiles. Only the last active tile may contain masked lanes. BM/BN/BK and FP32
accumulation order match batch_invariant.linear, independent of route counts.
Scratch shapes are static; this does NOT evaluate all experts on all tokens.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _compact(IDS, WEIGHTS, TOKENS, COUNTS, ROUTE_W,
             T: tl.constexpr, TOPK: tl.constexpr, FIRST: tl.constexpr,
             BLOCK: tl.constexpr):
    e = tl.program_id(0)
    t = tl.arange(0, BLOCK)
    hit = tl.full((BLOCK,), False, tl.int1)
    weight = tl.full((BLOCK,), 0., tl.float32)
    for choice in tl.static_range(TOPK):
        idx = tl.load(IDS + t * TOPK + choice, t < T, other=-1)
        w = tl.load(WEIGHTS + t * TOPK + choice, t < T, other=0.)
        selected = (t < T) & (idx == FIRST + e)
        hit |= selected
        weight += tl.where(selected, w.to(tl.float32), 0.)
    position = tl.cumsum(hit.to(tl.int32)) - 1
    tl.store(TOKENS + e * T + position, t, hit)
    tl.store(COUNTS + e, tl.sum(hit.to(tl.int32), 0))
    tl.store(ROUTE_W + e * T + t, weight, t < T)


@triton.jit
def _gemm(X, W, TOKENS, COUNTS, ROUTE_W, Y,
          T: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          GATHER: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(2)
    start = tl.program_id(0) * BM
    count = tl.load(COUNTS + e)
    # A device branch is replayed on every invocation; capture does not freeze
    # the count. Empty expert tiles perform no weight loads or matrix products.
    if start < count:
        row = start + tl.arange(0, BM)
        col = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        token = tl.load(TOKENS + e * T + row, row < count, other=0)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + ks
            if GATHER:
                offsets = token[:, None] * K + kk[None, :]
            else:
                offsets = (e * T + row[:, None]) * K + kk[None, :]
            a = tl.load(X + offsets, (row[:, None] < count) & (kk[None, :] < K), other=0.)
            b = tl.load(W + (e * N + col[None, :]) * K + kk[:, None],
                        (col[None, :] < N) & (kk[:, None] < K), other=0.)
            acc = tl.dot(a, b, acc, input_precision='ieee')
        value = acc.to(Y.dtype.element_ty)
        if GATHER:
            destination = (e * T + row[:, None]) * N + col[None, :]
        else:
            w = tl.load(ROUTE_W + e * T + token, row < count, other=0.)
            # Preserve the BF16 down-projection boundary and BF16 route product.
            value = (value.to(tl.float32) * w[:, None]).to(Y.dtype.element_ty)
            destination = (e * T + token[:, None]) * N + col[None, :]
        tl.store(Y + destination, value,
                 (row[:, None] < count) & (col[None, :] < N))


@triton.jit
def _activation(H13, COUNTS, ACT, T: tl.constexpr, I: tl.constexpr,
                BLOCK: tl.constexpr):
    e = tl.program_id(1)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(COUNTS + e)
    if tl.program_id(0) * BLOCK < count * I:
        row, col = offset // I, offset % I
        mask = offset < count * I
        gate = tl.load(H13 + (e * T + row) * (2 * I) + col, mask, other=0.).to(tl.float32)
        up = tl.load(H13 + (e * T + row) * (2 * I) + I + col, mask, other=0.).to(tl.float32)
        silu = (gate / (1. + tl.exp(-gate))).to(ACT.dtype.element_ty).to(tl.float32)
        tl.store(ACT + e * T * I + offset, silu * up, mask)


def sparse_expert_routes(x, idx, weights, w13, w2, first_expert):
    """Return weighted [E_local,T,H] with zeros for unselected expert/token pairs.

    Sorting/compaction and both GEMMs stay on device. No .item(), nonzero,
    data-dependent tensor sizes, CPU split vectors, or atomic output sums.
    """
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError('sparse decode requires CUDA BF16/FP16 experts')
    if not all(t.is_contiguous() for t in (x, idx, weights, w13, w2)):
        raise ValueError('sparse decode requires contiguous inputs and weights')
    t, h = x.shape
    e, _, _ = w13.shape
    i = w2.shape[-1]
    if w13.shape != (e, 2 * i, h) or w2.shape != (e, h, i):
        raise ValueError('sparse decode expert weight shape mismatch')
    if idx.ndim != 2 or idx.shape[0] != t or weights.shape != idx.shape:
        raise ValueError('sparse decode route shape mismatch')
    out = x.new_zeros((e, t, h))
    if not t:
        return out
    tokens = torch.empty((e, t), device=x.device, dtype=torch.int32)
    counts = torch.empty(e, device=x.device, dtype=torch.int32)
    route_w = torch.empty((e, t), device=x.device, dtype=torch.float32)
    h13 = x.new_empty((e, t, 2 * i))
    act = x.new_empty((e, t, i))
    _compact[(e,)](idx, weights, tokens, counts, route_w, t, idx.shape[1], first_expert,
                   BLOCK=triton.next_power_of_2(t))
    options = dict(BM=16, BN=64, BK=32, num_warps=4, num_stages=2, enable_fp_fusion=False)
    _gemm[(triton.cdiv(t, 16), triton.cdiv(2 * i, 64), e)](
        x, w13, tokens, counts, route_w, h13, t, 2 * i, h, GATHER=True, **options)
    _activation[(triton.cdiv(t * i, 512), e)](
        h13, counts, act, t, i, BLOCK=512, enable_fp_fusion=False)
    _gemm[(triton.cdiv(t, 16), triton.cdiv(h, 64), e)](
        act, w2, tokens, counts, route_w, out, t, h, i, GATHER=False, **options)
    return out
