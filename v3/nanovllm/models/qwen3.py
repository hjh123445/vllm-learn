# v3/nanovllm/models/qwen3.py
"""v3 模型：v2 的实现（平铺 + context），适配 nano 的接口约定——
构造用 hf_config（AutoConfig），attention 从 get_context() 读上下文，
k_cache/v_cache 由 ModelRunner 外部注入。"""
import json
import os
import torch
import torch.nn as nn

from nanovllm.layers.linear import BaseLinear


def _rope_theta(config) -> float:
    """rope_theta：新版 transformers 把它挪进了 config.rope_parameters；直接读 config.json 最稳。"""
    if hasattr(config, "rope_theta"):
        return config.rope_theta
    rp = getattr(config, "rope_parameters", None) or {}
    if isinstance(rp, dict) and "rope_theta" in rp:
        return rp["rope_theta"]
    return json.load(open(os.path.join(config._name_or_path, "config.json")))["rope_theta"]
from nanovllm.layers.rotary_embedding import RotaryEmbeddding
from nanovllm.layers.norm import RmsNorm
from nanovllm.layers.attention import Attention
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.embedding import LmHead, Embedding


class QwenAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_head = config.num_attention_heads
        self.num_kv_head = config.num_key_value_heads
        self.q_size = self.num_head * self.head_dim
        self.kv_size = self.num_kv_head * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = BaseLinear(config.hidden_size, self.q_size)
        self.k_proj = BaseLinear(config.hidden_size, self.kv_size)
        self.v_proj = BaseLinear(config.hidden_size, self.kv_size)
        self.o_proj = BaseLinear(self.q_size, config.hidden_size)

        self.q_norm = RmsNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RmsNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = None      # 由 QwenModel 注入共享
        self.attn = Attention(self.num_head, self.num_kv_head, self.head_dim, self.scale)

    def forward(self, x, positions):
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.num_head, self.head_dim)
        k = self.k_proj(x).view(T, self.num_kv_head, self.head_dim)
        v = self.v_proj(x).view(T, self.num_kv_head, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(positions, q, k)
        o = self.attn(q, k, v)                   # [total_q, num_head*head_dim]
        return self.o_proj(o)


class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_act == "silu"
        self.fc1 = BaseLinear(config.hidden_size, config.intermediate_size * 2)
        self.fc2 = BaseLinear(config.intermediate_size, config.hidden_size)
        self.act = SiluAndMul()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class QwenDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = QwenAttention(config, layer_idx)
        self.mlp = QwenMLP(config)
        self.input_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, positions):
        residual = x
        x = self.self_attn(self.input_layernorm(x), positions)
        x = x + residual
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = x + residual
        return x


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
        for layer in self.layers:
            x = layer(x, positions)
        return self.norm(x)


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
