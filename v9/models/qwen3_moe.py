# v6/models/qwen3_moe.py
"""v6 模型：Qwen3-Omni 的 thinker（文本 LLM 部分，MoE 架构）。

attention 栈沿用 v4/v5（qk_norm + rope + GQA + 平铺 context + triton flash），
MLP 换成 MoE（router + grouped experts，layers/moe.py）。
只做文本路径：vision/audio tower 的权重 loader 直接跳过。"""
import json
import os
import torch
import torch.nn as nn

from layers.linear import BaseLinear, ColumnParallelLinear, RowParallelLinear
from layers.embedding import Embedding, LmHead
from layers.rotary_embedding import RotaryEmbeddding
from layers.norm import RmsNorm
from layers.attention import Attention
from layers.kernels import fused_add_rmsnorm
from layers.moe import MoEMLP


class Qwen3MoeAttention(nn.Module):
    """与 v4 的 QwenAttention 同构（qk_norm + GQA），平铺输入。

    v9.3 并行组件化：q/k/v 用 ColumnParallelLinear（按头切），o_proj 用
    RowParallelLinear（allreduce 内置）。构造时从 ParallelState 读 tp——
    本 rank 的头数/权重形状直接定型，loader 只装本 rank 切片。
    GQA 对齐保证：q 头 j 映射 kv 头 j//(H/kv)，q 的均分块恰好落在
    kv 的（可能复制的）切片上——kv_heads < tp 时多 rank 共享同一 kv 头。"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        # per-rank 头数（与组件的 shard_bounds 同数学）
        from layers.parallel import shard_bounds, STATE
        _, q_len = shard_bounds(config.num_attention_heads * self.head_dim,
                                STATE.tp_size, STATE.tp_rank)
        _, kv_len = shard_bounds(config.num_key_value_heads * self.head_dim,
                                 STATE.tp_size, STATE.tp_rank, min_out=self.head_dim)
        self.num_head = q_len // self.head_dim
        self.num_kv_head = kv_len // self.head_dim
        self.q_size = self.num_head * self.head_dim            # 本 rank 的 q 维
        self.kv_size = self.num_kv_head * self.head_dim

        self.q_proj = ColumnParallelLinear(config.hidden_size,
                                           config.num_attention_heads * self.head_dim)
        # k/v: min_out=head_dim —— kv 头不足 tp 时按头复制（shard_bounds 内处理）
        self.k_proj = ColumnParallelLinear(config.hidden_size,
                                           config.num_key_value_heads * self.head_dim,
                                           min_out=self.head_dim)
        self.v_proj = ColumnParallelLinear(config.hidden_size,
                                           config.num_key_value_heads * self.head_dim,
                                           min_out=self.head_dim)
        # in_dim 用全量 q 维（组件内部切 1/tp，forward 的输入是本 rank 切片）
        self.o_proj = RowParallelLinear(config.num_attention_heads * self.head_dim,
                                        config.hidden_size)

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
        return self.o_proj(o.reshape(T, self.q_size))    # allreduce 在组件内


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
        x = self.self_attn(x, positions)     # o_proj(RowParallel) 内置 allreduce
        x, residual = fused_add_rmsnorm(x, residual, self.post_attention_layernorm.weight,
                                       self.post_attention_layernorm.eps)
        x = self.mlp(x)                      # MoE 部分和聚合在 MoEMLP 内
        return x, residual


class Qwen3MoeModel(nn.Module):
    def __init__(self, config, max_position: int):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
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
        self.lm_head = LmHead(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, x):
        return self.lm_head(x)

    def quantize_int4(self, group: int = 128, skip_dense: bool = False):
        """就地量化（装载后调用）：
        - 专家权重（GroupedExperts.quantize_int4）——显存收益主体（~80% 权重）
        - dense 线性（q/k/v/o_proj，skip_dense=True 时跳过）——BaseLinear 的
          q/o 权重小（<2%），量化收益小且和 attention 的 triton kernel 耦合，
          本版只量化 k/v 之外的独立 Linear 由 QuantizedLinear 替换
        lm_head/embedding/router 不量化（vocab 维敏感 + 占比小）。"""
        import torch as _t
        from layers.quantized import QuantizedLinear
        n_replaced = 0
        for layer in self.model.layers:
            experts = layer.mlp.experts
            if hasattr(experts, "quantize_int4"):
                experts.quantize_int4(group)
                n_replaced += 1
        # dense: q_proj/k_proj/v_proj/o_proj → QuantizedLinear（B1 路径）
        # v9.3: 投影已是并行组件——按【本 rank 的形状】量化（粒度沿输入维 K，
        # 与输出维切分正交；DSV4 DESIGN §II.3.2 的"切分后再量化"约定）
        if not skip_dense:
            for layer in self.model.layers:
                attn = layer.self_attn
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    lin = getattr(attn, name)
                    w = lin.weight.data                        # [本rankN, K] 或 [N, 本rankK]
                    ql = QuantizedLinear(w.shape[1], w.shape[0], group)
                    ql.weight_loader(None, w)
                    ql = ql.to(w.device)
                    setattr(attn, name, ql)
                    del lin
        _t.cuda.empty_cache()
        return n_replaced
