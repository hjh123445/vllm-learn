"""Opt-in fixed-reduction math for reproducible decode batch buckets.

Unlike a global aten override, these functions are called only by our linear
and expert components. Tiles and K order never depend on token count; collectives
move rank contributions without reducing them, then sum in ascending rank order.
This follows vLLM's batch-invariance design, with a deliberately small kernel.
It costs communication/scratch memory and is not a throughput optimization.
"""
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl


def enabled():
    return os.getenv('VLLM_LEARN_BATCH_INVARIANT', '0') == '1'


@triton.jit
def _fixed_matmul(X, W, Y, BIAS, M, N: tl.constexpr, K: tl.constexpr,
                  XE, XM, XK, WE, WN, WK, HAS_BIAS: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(2)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + ks
        a = tl.load(X + e * XE + rows[:, None] * XM + kk[None, :] * XK,
                    mask=(rows[:, None] < M) & (kk[None, :] < K), other=0.)
        b = tl.load(W + e * WE + cols[None, :] * WN + kk[:, None] * WK,
                    mask=(cols[None, :] < N) & (kk[:, None] < K), other=0.)
        acc = tl.dot(a, b, acc, input_precision='ieee')
    if HAS_BIAS:
        acc += tl.load(BIAS + cols, mask=cols < N, other=0.)[None, :]
    tl.store(Y + (e * M + rows[:, None]) * N + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def linear(x, weight, bias=None):
    """[M,K] @ [N,K].T, or [E,M,K] @ [E,N,K].T."""
    if not enabled() or not x.is_cuda:
        if x.ndim == 2:
            return F.linear(x, weight, bias)
        return torch.bmm(x, weight.transpose(-2, -1))
    if x.ndim not in (2, 3) or weight.ndim != x.ndim:
        raise ValueError('fixed matmul requires matching 2D or 3D inputs')
    if x.dtype != weight.dtype or x.shape[-1] != weight.shape[-1]:
        raise ValueError('fixed matmul dtype/K mismatch')
    if x.ndim == 3 and (bias is not None or x.shape[0] != weight.shape[0]):
        raise ValueError('batched fixed matmul requires equal expert counts and no bias')
    m, k, n = x.shape[-2], x.shape[-1], weight.shape[-2]
    experts = x.shape[0] if x.ndim == 3 else 1
    out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    if m:
        _fixed_matmul[(triton.cdiv(m, 16), triton.cdiv(n, 64), experts)](
            x, weight, out, bias if bias is not None else out, m, n, k,
            x.stride(0) if x.ndim == 3 else 0, *x.stride()[-2:],
            weight.stride(0) if weight.ndim == 3 else 0, *weight.stride()[-2:],
            HAS_BIAS=bias is not None, BM=16, BN=64, BK=32,
            num_warps=4, num_stages=2, enable_fp_fusion=False)
    return out


@triton.jit
def _rank_sum(SRC, OUT, SIZE, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0., tl.float32)
    for rank in tl.static_range(WORLD):
        value += tl.load(SRC + rank * SIZE.to(tl.int64) + offset,
                         mask=offset < SIZE, other=0.).to(tl.float32)
    tl.store(OUT + offset, value, mask=offset < SIZE)


def all_reduce(x, group):
    """Gather is pure data movement; FP32 sum has a fixed rank order."""
    world = dist.get_world_size(group)
    if world == 1 or x.numel() == 0:
        return x
    value = x.contiguous()
    count = value.numel()
    gathered = torch.empty((world, count), device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(gathered, value.view(-1), group=group)
    output = torch.empty_like(value)
    _rank_sum[(triton.cdiv(count, 1024),)](gathered, output, count,
                                        WORLD=world, BLOCK=1024, enable_fp_fusion=False)
    return output


def expert_sum(x):
    """[E,T,H] -> [T,H], with ascending expert order independent of T."""
    value = x.contiguous()
    output = torch.empty(value.shape[1:], device=x.device, dtype=torch.float32)
    count = output.numel()
    if count:
        _rank_sum[(triton.cdiv(count, 1024),)](value, output, count,
            WORLD=value.shape[0], BLOCK=1024, enable_fp_fusion=False)
    return output
