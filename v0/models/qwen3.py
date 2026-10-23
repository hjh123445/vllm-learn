import torch
import torch.nn as nn

from config import Qwen3Config
from layers.linear import BaseLinear
from layers.rotary_embedding import RotaryEmbeddding
from layers.norm import RmsNorm
from layers.attention import Attention
from layers.activation import SiluAndMul
from layers.embedding import LmHead, Embedding


class QwenAttention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_head = config.num_attention_heads
        self.num_kv_head = config.num_key_value_heads
        self.q_size = self.num_head * self.head_dim
        self.kv_size = self.num_kv_head * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = BaseLinear(config.hidden_size, self.q_size)
        self.k_proj = BaseLinear(config.hidden_size, self.kv_size)
        self.v_proj = BaseLinear(config.hidden_size, self.kv_size)
        self.o_proj = BaseLinear(self.q_size, config.hidden_size)

        self.q_norm = RmsNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RmsNorm(config.head_dim, eps=config.rms_norm_eps)
        # RoPE 由 QwenModel 统一建一份再注入共享，避免 28 层各存一份 cos_sin cache
        self.rope = None
        # KV cache / RoPE 容量都用引擎的 max_seq_len（如 4096），
        # 而不是 config.max_position_embeddings（40960），否则 cache 白占 10 倍显存
        self.attn = Attention(batch_size=config.batch_size,
                              max_position=config.max_seq_len,
                              num_head=self.num_head,
                              num_kv_head=self.num_kv_head,
                              head_dim=self.head_dim,
                              scale=self.scale)

    def forward(self, x, positions, attn_mask=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(positions=positions, q=q, k=k)
        o = self.attn(q, k, v, attn_mask)                         # [b, num_head, s, head_dim]
        return self.o_proj(o.transpose(1, 2).flatten(2))          # [b, s, hidden]


class QwenMLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        assert config.hidden_act == "silu"
        self.fc1 = BaseLinear(config.hidden_size, config.intermediate_size * 2)
        self.fc2 = BaseLinear(config.intermediate_size, config.hidden_size)
        self.act = SiluAndMul()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, x, attn_mask=None):
        residual = x
        x = self.self_attn(self.input_layernorm(x), positions, attn_mask)
        x = x + residual
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = x + residual
        return x


class QwenModel(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 全模型共享一份 RoPE 表
        rope = RotaryEmbeddding(max_position=config.max_seq_len,
                                base=config.rope_theta,
                                head_dim=config.head_dim)
        for layer in self.layers:
            layer.self_attn.rope = rope

    def forward(self, input_ids, positions, attn_mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(positions, x, attn_mask)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):                                # 类名对齐HF
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = QwenModel(config)
        self.lm_head = LmHead(config.hidden_size, config.vocab_size)
        if config.tie_word_embeddings:                            # tie 在这里处理
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions, attn_mask=None):
        return self.model(input_ids, positions, attn_mask)

    def compute_logits(self, x):
        return self.lm_head(x)
