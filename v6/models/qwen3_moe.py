# v6/models/qwen3_moe.py
"""v6 模型：Qwen3-Omni 的 thinker（文本 LLM 部分，MoE 架构）。

attention 栈沿用 v4/v5（qk_norm + rope + GQA + 平铺 context + triton flash），
MLP 换成 MoE（router + grouped experts，layers/moe.py）。
只做文本路径：vision/audio tower 的权重 loader 直接跳过。"""
import json
import os
import torch
import torch.nn as nn

from layers.linear import BaseLinear
from layers.rotary_embedding import RotaryEmbeddding
from layers.norm import RmsNorm
from layers.attention import Attention
from layers.kernels import fused_add_rmsnorm
from layers.moe import MoEMLP


class Qwen3MoeAttention(nn.Module):
    """与 v4 的 QwenAttention 同构（qk_norm + GQA），平铺输入。"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_head = config.num_attention_heads
        self.num_kv_head = config.num_key_value_heads
        self.q_size = self.num_head * self.head_dim
        self.kv_size = self.num_kv_head * self.head_dim

        self.q_proj = BaseLinear(config.hidden_size, self.q_size)
        self.k_proj = BaseLinear(config.hidden_size, self.kv_size)
        self.v_proj = BaseLinear(config.hidden_size, self.kv_size)
        self.o_proj = BaseLinear(self.q_size, config.hidden_size)

        self.q_norm = RmsNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RmsNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = None
        self.attn = Attention(self.num_head, self.num_kv_head, self.head_dim,
                              self.head_dim ** -0.5)

    def forward(self, x, positions):
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.num_head, self.head_dim)
        k = self.k_proj(x).view(T, self.num_kv_head, self.head_dim)
        v = self.v_proj(x).view(T, self.num_kv_head, self.head_dim)
        o = self.attn(q, k, v, positions, self.q_norm.weight, self.k_norm.weight,
                      self.rope.cos_sin_cache, self.q_norm.eps)
        return self.o_proj(o.reshape(T, self.q_size))


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        self.mlp = MoEMLP(config)                       # 全部 48 层都是 MoE（sparse_step=1）
        self.input_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, residual, positions):
        x, residual = fused_add_rmsnorm(x, residual, self.input_layernorm.weight,
                                       self.input_layernorm.eps)
        x = self.self_attn(x, positions)
        x, residual = fused_add_rmsnorm(x, residual, self.post_attention_layernorm.weight,
                                       self.post_attention_layernorm.eps)
        return self.mlp(x), residual


class Qwen3MoeModel(nn.Module):
    def __init__(self, config, max_position: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        rope = RotaryEmbeddding(max_position=max_position,
                                base=config.rope_theta,
                                head_dim=config.head_dim)
        for layer in self.layers:
            layer.self_attn.rope = rope

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)
        residual = torch.zeros_like(x)
        for layer in self.layers:
            x, residual = layer(x, residual, positions)
        out, _ = fused_add_rmsnorm(x, residual, self.norm.weight, self.norm.eps)
        return out


class Qwen3MoeForCausalLM(nn.Module):
    def __init__(self, config, max_position: int):
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config, max_position)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, x):
        return self.lm_head(x)
