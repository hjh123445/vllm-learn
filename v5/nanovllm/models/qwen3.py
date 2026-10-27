# v4/nanovllm/models/qwen3.py
"""v4 模型：v3 的结构 + 融合算子接线。

残差管道改为 vLLM 风格：每层的残差加法融进下一个 rmsnorm kernel，
全模型 forward 里零 aten add。MLP 的 silu*mul 换 triton kernel。"""
import json
import os
import torch
import torch.nn as nn

from nanovllm.layers.linear import BaseLinear
from nanovllm.layers.rotary_embedding import RotaryEmbeddding
from nanovllm.layers.norm import RmsNorm
from nanovllm.layers.attention import Attention
from nanovllm.layers.kernels import fused_add_rmsnorm, silu_and_mul
from nanovllm.layers.embedding import LmHead, Embedding


def _rope_theta(config) -> float:
    """rope_theta：新版 transformers 把它挪进了 config.rope_parameters；直接读 config.json 最稳。"""
    if hasattr(config, "rope_theta"):
        return config.rope_theta
    rp = getattr(config, "rope_parameters", None) or {}
    if isinstance(rp, dict) and "rope_theta" in rp:
        return rp["rope_theta"]
    return json.load(open(os.path.join(config._name_or_path, "config.json")))["rope_theta"]


class QwenAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
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
        self.rope = None      # 由 QwenModel 注入共享
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


class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_act == "silu"
        self.fc1 = BaseLinear(config.hidden_size, config.intermediate_size * 2)
        self.fc2 = BaseLinear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.fc2(silu_and_mul(self.fc1(x)))


class QwenDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = QwenAttention(config, layer_idx)
        self.mlp = QwenMLP(config)
        self.input_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, residual, positions):
        """残差加法融进下一个 rmsnorm：本层返回 (未归一化输出, 残差)，
        加法由下一层的 fused_add_rmsnorm 消费。"""
        x, residual = fused_add_rmsnorm(x, residual, self.input_layernorm.weight,
                                       self.input_layernorm.eps)
        x = self.self_attn(x, positions)
        x, residual = fused_add_rmsnorm(x, residual, self.post_attention_layernorm.weight,
                                       self.post_attention_layernorm.eps)
        return self.mlp(x), residual


class QwenModel(nn.Module):
    def __init__(self, config, max_position: int):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        rope = RotaryEmbeddding(max_position=max_position,
                                base=_rope_theta(config),
                                head_dim=getattr(config, "head_dim",
                                                  config.hidden_size // config.num_attention_heads))
        for layer in self.layers:
            layer.self_attn.rope = rope

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)                   # [total_q, hidden]
        residual = torch.zeros_like(x)
        for layer in self.layers:
            x, residual = layer(x, residual, positions)
        out, _ = fused_add_rmsnorm(x, residual, self.norm.weight, self.norm.eps)
        return out


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config, max_position: int):
        super().__init__()
        self.config = config
        self.model = QwenModel(config, max_position)
        self.lm_head = LmHead(config.hidden_size, config.vocab_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions):
        """模型只见 input_ids 和 positions——attention 上下文在全局 Context 里。"""
        return self.model(input_ids, positions)

    def compute_logits(self, x):
        return self.lm_head(x)
