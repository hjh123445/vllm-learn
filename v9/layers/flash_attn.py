# v5/layers/flash_attn.py
"""v4 手写 triton FlashAttention（直接注意力，替换 v3 的纯 PyTorch gather + matmul + softmax）。

两个 kernel：
  decode: paged flash decoding——q 每头一路在线 softmax，K/V 按 block 表逐块从
          cache 直读（零 gather 零 repeat_interleave），GQA 头组映射在 kernel 内做
  prefill: paged causal flash——支持 chunked prefill / 前缀缓存（q 是 k 的尾段，
          causal 底右对齐：query 绝对位置 = seqlen_k - seqlen_q + j）

数值约定：全部 load 后升 fp32 计算（decode 用逐元素乘加，prefill 用 tl.dot ieee），
fp32 输入下与纯 PyTorch 路径逐位一致（accuracy T4 依赖这一点）。"""
import torch
import triton
import triton.language as tl


# ================= paged flash decoding =================

@triton.jit
def _decode_attn_kernel(Q, KC, VC, BT, CTX, OUT, SCALE,
                        HK: tl.constexpr, HD: tl.constexpr,
                        BS: tl.constexpr, BK: tl.constexpr, MAXB,
                        BT_ROW_STRIDE):
    """grid = (n, num_heads)。每 program 一个 (序列, q头)：
    在线 softmax 流式扫自己的 KV block，acc 累加 @ v。"""
    n = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // (tl.num_programs(1) // HK)
    ctx = tl.load(CTX + n)

    d = tl.arange(0, HD)
    q = tl.load(Q + n.to(tl.int64) * tl.num_programs(1) * HD + h * HD + d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HD], dtype=tl.float32)

    for r0 in range(0, ctx, BK):
        rows = r0 + tl.arange(0, BK)
        valid = rows < ctx
        blk = tl.load(BT + n.to(tl.int64) * BT_ROW_STRIDE + rows // BS,
                      mask=valid, other=0).to(tl.int64)
        # cache 布局 [blocks, BS, HK, HD]，全局槽位 = blk*BS + rows%BS
        off = (blk * BS + rows % BS)[:, None] * (HK * HD) + hk * HD + d[None, :]
        k = tl.load(KC + off, mask=valid[:, None], other=0.0).to(tl.float32)   # [BK, HD]
        v = tl.load(VC + off, mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, 1) * SCALE                                  # [BK]
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 0))
        alpha = tl.math.exp(m_i - m_new)
        p = tl.math.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        m_i = m_new

    out = acc / l_i
    tl.store(OUT + n.to(tl.int64) * tl.num_programs(1) * HD + h * HD + d,
             out.to(OUT.dtype.element_ty))


def decode_attn(q: torch.Tensor, k_cache, v_cache, block_tables, context_lens,
                num_kv_heads: int, scale: float):
    """q [n, num_heads, hd]（已 rope）→ out [n, num_heads*hd]。"""
    n, HQ, HD = q.shape
    BS = k_cache.shape[1]
    out = torch.empty_like(q)
    _decode_attn_kernel[(n, HQ)](
        q, k_cache, v_cache, block_tables, context_lens, out, scale,
        HK=num_kv_heads, HD=HD, BS=BS, BK=32, MAXB=block_tables.shape[1],
        BT_ROW_STRIDE=block_tables.stride(0), num_warps=4)
    return out.reshape(n, HQ * HD)


# ================= paged causal flash prefill =================

@triton.jit
def _prefill_attn_kernel(Q, KC, VC, BT, CU_Q, CU_K, OUT, SCALE,
                         HK: tl.constexpr, HD: tl.constexpr,
                         BS: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                         MAXB, BT_ROW_STRIDE):
    """grid = (num_seqs, num_q_blocks, num_heads)。
    causal 底右对齐：query 绝对位置 = seqlen_k - seqlen_q + j，
    key 槽位 = 绝对位置（chunked prefill / 前缀缓存通用）。"""
    pid_seq = tl.program_id(0)
    pid_m = tl.program_id(1)
    h = tl.program_id(2)
    HQ = tl.num_programs(2)
    hk = h // (HQ // HK)
    q0 = tl.load(CU_Q + pid_seq)
    q1 = tl.load(CU_Q + pid_seq + 1)
    # cu_seqlens 是绝对前缀和：键长 = 本序列终点 - 起点（含缓存前缀）
    k_len = tl.load(CU_K + pid_seq + 1) - tl.load(CU_K + pid_seq)
    sq = q1 - q0

    d = tl.arange(0, HD)
    m_rows = pid_m * BM + tl.arange(0, BM)
    valid_q = m_rows < sq
    qm = q0 + m_rows                              # flat 批里的行号
    q = tl.load(Q + qm.to(tl.int64)[:, None] * HQ * HD + h * HD + d[None, :],
                mask=valid_q[:, None], other=0.0)      # [BM, HD]（保持原 dtype，dot 用）
    abs_pos = k_len - sq + m_rows                 # query 绝对位置

    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, HD], dtype=tl.float32)

    for r0 in range(0, k_len, BK):
        rows = r0 + tl.arange(0, BK)
        valid_k = rows < k_len
        blk = tl.load(BT + pid_seq.to(tl.int64) * BT_ROW_STRIDE + rows // BS,
                      mask=valid_k, other=0).to(tl.int64)
        off = (blk * BS + rows % BS)[:, None] * (HK * HD) + hk * HD + d[None, :]
        k = tl.load(KC + off, mask=valid_k[:, None], other=0.0)   # [BK, HD] 原 dtype
        v = tl.load(VC + off, mask=valid_k[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE   # [BM, BK] fp32
        causal = abs_pos[:, None] >= rows[None, :]
        s = tl.where(causal & valid_k[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp(m_i - m_new)
        p = tl.math.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_new

    o = acc / l_i[:, None]
    tl.store(OUT + qm.to(tl.int64)[:, None] * HQ * HD + h * HD + d[None, :],
             o.to(OUT.dtype.element_ty), mask=valid_q[:, None])


def prefill_attn(q: torch.Tensor, k_cache, v_cache, block_tables,
                 cu_seqlens_q, cu_seqlens_k, max_seqlen_q: int,
                 num_kv_heads: int, scale: float):
    """q [total_q, num_heads, hd]（已 rope）→ out [total_q, num_heads*hd]。"""
    T, HQ, HD = q.shape
    BS = k_cache.shape[1]
    n = cu_seqlens_q.shape[0] - 1
    out = torch.empty_like(q)
    grid = (n, triton.cdiv(max_seqlen_q, 32), HQ)
    _prefill_attn_kernel[grid](
        q, k_cache, v_cache, block_tables, cu_seqlens_q, cu_seqlens_k, out, scale,
        HK=num_kv_heads, HD=HD, BS=BS, BM=32, BK=32,
        MAXB=block_tables.shape[1], BT_ROW_STRIDE=block_tables.stride(0),
        num_warps=4)
    return out.reshape(T, HQ * HD)
