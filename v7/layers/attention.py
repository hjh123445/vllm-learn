# v5/layers/attention.py
"""v4 attention：cache 写入 + 范数 + rope 融成一个 kernel（kernels.py），
直接注意力换成手写 triton FlashAttention（flash_attn.py）。
纯 PyTorch 路径保留为 fallback（fp32 数值交叉验证用）。"""
import torch
from torch import nn

from utils.context import get_context
from layers.kernels import fused_qk_norm_rope_store
from layers.flash_attn import decode_attn, prefill_attn


class Attention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.k_cache = self.v_cache = torch.tensor([])   # ModelRunner 分配后指过来

    def forward(self, q, k, v, positions, q_norm_w, k_norm_w, cos_sin_cache, eps):
        """q/k/v: [total_q, heads, head_dim]（投影输出，未 norm 未 rope）。
        这里完成 norm + rope + 写 cache + 注意力，全部 triton。"""
        ctx = get_context()
        q = fused_qk_norm_rope_store(q, k, v, q_norm_w, k_norm_w, cos_sin_cache,
                                     positions, ctx.slot_mapping,
                                     self.k_cache, self.v_cache, eps)
        if ctx.is_prefill:
            return prefill_attn(q, self.k_cache, self.v_cache, ctx.block_tables,
                                 ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                                 ctx.max_seqlen_q, self.num_kv_heads, self.scale)
        return decode_attn(q, self.k_cache, self.v_cache, ctx.block_tables,
                           ctx.context_lens, self.num_kv_heads, self.scale)
