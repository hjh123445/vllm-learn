# v3/nanovllm/layers/attention.py
"""v3 attention：读全局 Context（nano 的接口约定），纯 PyTorch 实现。

与 nano 的差异只在 kernel：nano 用 flash_attn_varlen_func / flash_attn_with_kvcache +
triton store_kvcache；我们用 index_put 写 cache + 逐序列/varlen 语义的朴素注意力。
k_cache/v_cache 缓冲区由 ModelRunner 分配后指进来（[num_blocks, block_size, kvh, hd]，
全局槽位 = block_id * block_size + 块内偏移，可平铺索引）。"""
import torch
import torch.nn as nn

from nanovllm.utils.context import get_context

# 全局加性 causal mask 表（懒加载，全模型共享）：
# prefill 的每序列 mask 是它的一个切片（视图，零拷贝），不再每层每序列重建
_MASK_TABLE = None


def _mask_table(max_len: int, device, dtype):
    global _MASK_TABLE
    import torch
    if _MASK_TABLE is None or _MASK_TABLE.shape[0] < max_len or _MASK_TABLE.device != device:
        pos = torch.arange(max_len)
        causal = pos[:, None] >= pos[None, :]
        _MASK_TABLE = torch.where(causal, 0.0, torch.finfo(dtype).min).to(dtype).to(device)
    return _MASK_TABLE


class Attention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.k_cache = self.v_cache = torch.tensor([])   # ModelRunner 分配后指过来

    def _gather_seq(self, tabs, length):
        """一个序列的 cache → [1, kvh, length, hd]（gather 只搬需要的 block）。"""
        lay_k = self.k_cache[tabs]                       # [nblk, bs, kvh, hd]
        lay_v = self.v_cache[tabs]
        nblk, bs, H, D = lay_k.shape
        k = lay_k.permute(2, 0, 1, 3).reshape(H, nblk * bs, D)[:, :length]
        v = lay_v.permute(2, 0, 1, 3).reshape(H, nblk * bs, D)[:, :length]
        return k.unsqueeze(0), v.unsqueeze(0)

    def _gather_batch(self, tabs, read_len):
        """decode 组整组 gather → [n, kvh, read_len, hd]（一次拷贝，热路径）。"""
        lay_k = self.k_cache[tabs]                       # [n, nblk, bs, kvh, hd]
        lay_v = self.v_cache[tabs]
        n, nblk, bs, H, D = lay_k.shape
        k = lay_k.permute(0, 3, 1, 2, 4).reshape(n, H, nblk * bs, D)[:, :, :read_len]
        v = lay_v.permute(0, 3, 1, 2, 4).reshape(n, H, nblk * bs, D)[:, :, :read_len]
        return k, v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """q/k/v: [total_q, heads, head_dim]（已 rope，平铺无 b/s 维）。
        返回 [total_q, num_heads * head_dim]。"""
        ctx = get_context()
        T = q.shape[0]
        n_rep = self.num_heads // self.num_kv_heads

        # 写 cache：全局槽位一次 index_put（nano 用 triton store_kvcache，等价）
        if self.k_cache.numel():
            slots = ctx.slot_mapping.long()
            self.k_cache.view(-1, self.num_kv_heads, self.head_dim)[slots] = k
            self.v_cache.view(-1, self.num_kv_heads, self.head_dim)[slots] = v

        out = q.new_empty(T, self.num_heads * self.head_dim)

        if ctx.is_prefill:
            # prefill：逐序列（cu_seqlens 切分，chunked prefill / 前缀缓存时 q 是 k 的尾段，
            # causal 按"底右对齐"：query 绝对位置 = start + j，start = seqlen_k - seqlen_q）
            for i in range(len(ctx.cu_seqlens_q) - 1):
                i0, i1 = int(ctx.cu_seqlens_q[i]), int(ctx.cu_seqlens_q[i + 1])
                k_len = int(ctx.cu_seqlens_k[i + 1]) - int(ctx.cu_seqlens_k[i])
                start = k_len - (i1 - i0)
                tabs = ctx.block_tables[i]
                qi = q[i0:i1].unsqueeze(0).transpose(1, 2)          # [1, h, sq, hd]
                ki, vi = self._gather_seq(tabs, k_len)               # [1, kvh, sk, hd]
                ki = ki.repeat_interleave(n_rep, dim=1)
                vi = vi.repeat_interleave(n_rep, dim=1)
                scores = (qi @ ki.transpose(-2, -1)) * self.scale     # [1, h, sq, sk]
                # causal 切片：query 绝对位置 start..start+sq-1，key 槽位 0..k_len-1
                mask = _mask_table(max(start + (i1 - i0), k_len), q.device, q.dtype)[start:start + (i1 - i0), :k_len]
                scores = scores + mask[None, None]
                o = scores.softmax(-1) @ vi                           # [1, h, sq, hd]
                out[i0:i1] = o.transpose(1, 2).reshape(i1 - i0, -1)
        else:
            # decode：整组一次 gather + 一条批注意力（热路径）
            n = ctx.context_lens.shape[0]
            read_len = int(ctx.context_lens.max())
            qd = q.unsqueeze(2)                                       # [n, h, 1, hd]
            kd, vd = self._gather_batch(ctx.block_tables, read_len)   # [n, kvh, L, hd]
            kd = kd.repeat_interleave(n_rep, dim=1)
            vd = vd.repeat_interleave(n_rep, dim=1)
            scores = (qd @ kd.transpose(-2, -1)) * self.scale         # [n, h, 1, L]
            valid = torch.arange(read_len, device=q.device)[None, :] < ctx.context_lens[:, None]
            mask = torch.where(valid[:, None, None], 0.0, torch.finfo(q.dtype).min).to(q.dtype)
            scores = scores + mask
            od = scores.softmax(-1) @ vd                              # [n, h, 1, hd]
            out.copy_(od.reshape(n, -1))
        return out
