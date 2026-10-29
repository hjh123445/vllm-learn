"""v6.3 数值排查：单层对拍——我们的 DSV4DecoderLayer vs HF DeepseekV4DecoderLayer。

同权重同输入逐层对比 hidden，从分歧点定位 bug（v0 时代的标准方法）。
只跑 1 层（4 层已装到 cuda:0 的量级）——用 HF 的单层模块（不需要
全模型 584GB）。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/data/ifly_jyhe23/hot_model_top5/audio_understand/qwen2-audio/transformers-5.12.0/src")

MODEL = "/data/xmhuang6/deepseek/DeepSeek-V4-Flash"

# ---- 构造 HF 的单层（CPU，小规模 config 避免全专家）----
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer, DeepseekV4Attention)
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

import json
with open(os.path.join(MODEL, "config.json")) as f:
    raw = json.load(f)

# 缩小版 config（结构同、专家数少）——单层对拍不需要全 256 专家
small = dict(raw)
small.update(num_hidden_layers=4, n_routed_experts=4, num_experts_per_tok=2,
             vocab_size=32000)
cfg = DeepseekV4Config(**small)
print("HF config ok:", cfg.num_hidden_layers, "layers,", cfg.n_routed_experts, "experts")

torch.manual_seed(0)
hf_layer = DeepseekV4DecoderLayer(cfg, 3).float().eval()  # layer 3 = 非 hash 层
# HF 的 nn.Parameter 多为 empty（含 NaN 垃圾）——统一重初始化为有限随机值
for n, p_ in hf_layer.named_parameters():
    if not torch.isfinite(p_).all():
        torch.nn.init.normal_(p_, std=0.02)
for n, b_ in hf_layer.named_buffers():
    if b_.dtype.is_floating_point and not torch.isfinite(b_).all():
        torch.nn.init.normal_(b_, std=0.02)

# ---- 我们的层（同结构）----
from models.deepseek_v4 import DSV4DecoderLayer
from models.deepseek_v4 import DSV4Rope

my = DSV4DecoderLayer({
    "num_attention_heads": cfg.num_attention_heads,
    "head_dim": cfg.head_dim,
    "qk_rope_head_dim": cfg.qk_rope_head_dim,
    "q_lora_rank": cfg.q_lora_rank,
    "o_groups": cfg.o_groups,
    "o_lora_rank": cfg.o_lora_rank,
    "hidden_size": cfg.hidden_size,
    "rms_norm_eps": cfg.rms_norm_eps,
    "n_routed_experts": cfg.n_routed_experts,
    "moe_intermediate_size": cfg.moe_intermediate_size,
    "num_experts_per_tok": cfg.num_experts_per_tok,
    "num_hash_layers": cfg.default_num_hash_layers,
    "routed_scaling_factor": cfg.routed_scaling_factor,
    "hc_mult": cfg.hc_mult,
    "hc_sinkhorn_iters": cfg.hc_sinkhorn_iters,
    "vocab_size": cfg.vocab_size,
}, 3).float().eval()

# ---- 权重对齐（HF → 我们的）----
import torch.nn as nn


def copy_(dst, src):
    assert dst.shape == src.shape, (dst.shape, src.shape)
    dst.data.copy_(src.data)


# attention
attn_hf, attn_my = hf_layer.self_attn, my.self_attn
copy_(attn_my.q_a_proj.weight, attn_hf.q_a_proj.weight)
copy_(attn_my.q_a_norm.weight, attn_hf.q_a_norm.weight)
copy_(attn_my.q_b_proj.weight, attn_hf.q_b_proj.weight)
copy_(attn_my.kv_proj.weight, attn_hf.kv_proj.weight)
copy_(attn_my.kv_norm.weight, attn_hf.kv_norm.weight)
copy_(attn_my.o_a.weight, attn_hf.o_a_proj.weight)
copy_(attn_my.o_b.weight, attn_hf.o_b_proj.weight)
copy_(attn_my.sinks, attn_hf.sinks)
# norms
copy_(my.input_layernorm.weight, hf_layer.input_layernorm.weight)
copy_(my.post_attention_layernorm.weight, hf_layer.post_attention_layernorm.weight)
# MoE
mlp_hf, mlp_my = hf_layer.mlp, my.mlp
copy_(mlp_my.gate_weight, mlp_hf.gate.weight)
copy_(mlp_my.gate_bias, mlp_hf.gate.e_score_correction_bias)
# HF 的专家存 gate_up_proj 融合 [E, 2I, H] + down_proj [E, H, I]
gu = mlp_hf.experts.gate_up_proj        # [E, 2I, H]
for e in range(cfg.n_routed_experts):
    copy_(mlp_my.w1[e], gu[e, :cfg.moe_intermediate_size])          # gate = 前半
    copy_(mlp_my.w3[e], gu[e, cfg.moe_intermediate_size:])         # up = 后半
    copy_(mlp_my.w2[e], mlp_hf.experts.down_proj[e])
copy_(mlp_my.sw1, mlp_hf.shared_experts.gate_proj.weight)
copy_(mlp_my.sw3, mlp_hf.shared_experts.up_proj.weight)
copy_(mlp_my.sw2, mlp_hf.shared_experts.down_proj.weight)
# HC
for name in ("fn", "base", "scale"):
    copy_(getattr(my.attn_hc, name), getattr(hf_layer.attn_hc, name))
    copy_(getattr(my.ffn_hc, name), getattr(hf_layer.ffn_hc, name))
print("权重对齐完成")

# ---- 输入构造 ----
T, H, hc = 5, cfg.hidden_size, cfg.hc_mult
torch.manual_seed(42)
streams = torch.randn(T, hc, H)          # [T, hc, hidden]
input_ids = torch.randint(0, 32000, (T,))
positions = torch.arange(T)

# ---- HF 前向 ----
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    apply_rotary_pos_emb)

with torch.no_grad():
    # HF 层的 forward 需要 position_embeddings dict + input_ids
    rope = hf_layer.self_attn.rope if hasattr(hf_layer.self_attn, "rope") else None
    # 直接构造 main rope 的 cos/sin
    inv = 1.0 / (10000 ** (torch.arange(0, cfg.qk_rope_head_dim, 2).float()
                            / cfg.qk_rope_head_dim))
    freqs = torch.outer(positions.float(), inv)          # [T, rd/2]
    cos = freqs.cos().unsqueeze(0)                        # [B=1, T, rd/2]
    sin = freqs.sin().unsqueeze(0)
    pos_emb = {"main": (cos, sin), "compress": (cos, sin)}
    out_hf = hf_layer(streams.unsqueeze(0),  # [B=1, T, hc, H]
                      input_ids=input_ids.unsqueeze(0),
                      position_embeddings=pos_emb,
                      position_ids=positions.unsqueeze(0),
                      attention_mask=None)[0]
    out_hf = out_hf.squeeze(0)

    # 我们的前向
    # 单元对拍用标准 rope（yarn 与 HF 手推参考不是同一数学——在真实
    # 权重路径上验证 yarn）
    rope_my = DSV4Rope(cfg.head_dim, cfg.qk_rope_head_dim, base=10000,
                        yarn_factor=1.0)
    my_model_rope = rope_my
    my.self_attn.rope = rope_my
    out_my = my(streams, positions, input_ids)

print(f"out_hf: finite={torch.isfinite(out_hf).all().item()}, absmax={out_hf.abs().max().item():.4f}")
print(f"out_my: finite={torch.isfinite(out_my).all().item()}, absmax={out_my.abs().max().item():.4f}")
fin_h = torch.isfinite(out_hf) & torch.isfinite(out_my)
d = (out_hf[fin_h] - out_my[fin_h]).abs().max().item() if fin_h.any() else float("nan")
print(f"层输出 max diff (仅有限位置) = {d:.4e}")

# ---- 逐子模块下钻 ----
# 复现 HF 的中间量
with torch.no_grad():
    # HF: attn_hc
    post_hf, comb_hf, collapsed_hf = hf_layer.attn_hc(streams.unsqueeze(0))
    # 我们
    post_my, comb_my, collapsed_my = my.attn_hc(streams)
    print(f"attn_hc collapsed diff: {(collapsed_hf.squeeze(0) - collapsed_my).abs().max().item():.2e}")

    # ---- 我们的层逐步检查（定位 NaN 来源）----
    with torch.no_grad():
        s1 = my.input_layernorm(collapsed_my)
        print(f"after norm: finite={torch.isfinite(s1).all().item()}")
        o_my_dbg = my.self_attn(collapsed_my, positions)
        print(f"attn out: finite={torch.isfinite(o_my_dbg).all().item()}, "
              f"absmax={o_my_dbg.abs().max().item():.3f}")
        m_my_dbg = my.mlp(collapsed_my, input_ids)
        print(f"mlp out: finite={torch.isfinite(m_my_dbg).all().item()}, "
              f"absmax={m_my_dbg.abs().max().item():.3f}")

    # attention 子模块（HF 内部手推）
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb
    x = collapsed_hf.squeeze(0)                       # [T, H]
    qr = attn_hf.q_a_norm(attn_hf.q_a_proj(x))
    q = attn_hf.q_b_proj(qr).view(T, -1, attn_hf.num_heads, attn_hf.head_dim).transpose(1, 2)         if False else attn_hf.q_b_proj(qr).view(1, T, attn_hf.num_heads, attn_hf.head_dim).transpose(1, 2)
    q = attn_hf.q_b_norm(q)
    cos_b, sin_b = pos_emb["main"]
    q_r = apply_rotary_pos_emb(q, cos_b, sin_b)
    kv = attn_hf.kv_norm(attn_hf.kv_proj(x)).view(1, T, 1, attn_hf.head_dim).permute(0, 2, 1, 3)
    kv_r = apply_rotary_pos_emb(kv, cos_b, sin_b)

    # 我们的 attention 子模块（collapsed 用 HF 的——隔离 attention 差异）
    x2 = collapsed_hf.squeeze(0)
    qr2 = attn_my.q_a_norm(attn_my.q_a_proj(x2))
    q2 = attn_my.q_b_proj(qr2).view(T, attn_my.H, attn_my.D)
    q2 = attn_my._q_b_norm(q2)
    c2, s2 = rope_my.get(positions)
    q2r = attn_my.rope and None
    from layers.dsv4_components import apply_partial_rope
    q2r = apply_partial_rope(q2, c2, s2, attn_my.rd)
    print(f"q after rope diff: {(q_r.squeeze(0).permute(0,2,1) - q2r.transpose(1,2)).abs().max().item() if False else '手动比对见下'}")
    # 直接比 q: HF [1, H, T, D] vs ours [T, H, D]
    q_hf_flat = q_r.squeeze(0).permute(1, 0, 2)      # [T, H, D]
    print(f"q_rope diff: {(q_hf_flat - q2r).abs().max().item():.2e}")
    kv2 = attn_my.kv_norm(attn_my.kv_proj(x2))
    kv2r = apply_partial_rope(kv2.unsqueeze(1), c2, s2, attn_my.rd).squeeze(1)
    kv_hf_flat = kv_r.squeeze(0).squeeze(0)           # [T, D]
    print(f"kv_rope diff: {(kv_hf_flat - kv2r).abs().max().item():.2e}")

    # attention / MLP 输出（HF 手推 vs 我们的模块——同输入同权重）
    # HF 期望 [B, T, H]; 我们的平铺 [T, H]
    o_hf, _ = hf_layer.self_attn(collapsed_hf, position_embeddings=pos_emb,
                                  position_ids=positions.unsqueeze(0),
                                  attention_mask=None)
    o_my = attn_my(x2, positions)
    print(f"attn output diff: {(o_hf.squeeze(0) - o_my).abs().max().item():.2e}")

    m_hf = hf_layer.mlp(collapsed_hf, input_ids=input_ids.unsqueeze(0))
    m_my = my.mlp(x2, input_ids)
    print(f"mlp output diff: {(m_hf.squeeze(0) - m_my).abs().max().item():.2e}")

    # MoE 内部下钻
    with torch.no_grad():
        print(f"  gate_weight finite={torch.isfinite(my.mlp.gate_weight).all().item()}, "
              f"absmax={my.mlp.gate_weight.abs().max().item() if torch.isfinite(my.mlp.gate_weight).all() else float('nan'):.3f}")
        print(f"  gate_bias finite={torch.isfinite(my.mlp.gate_bias).all().item()}")
        logits = torch.nn.functional.linear(x2.float(), my.mlp.gate_weight.float())
        scores = torch.nn.functional.softplus(logits).sqrt()
        print(f"  scores finite={torch.isfinite(scores).all().item()}, "
              f"min={scores.min().item():.4f}")
        idx, w = my.mlp._route(x2, input_ids)
        print(f"  route idx={idx.tolist()}, w finite={torch.isfinite(w).all().item()}")
        # shared expert
        sh = torch.nn.functional.silu(x2 @ my.mlp.sw1.t()) * (x2 @ my.mlp.sw3.t())
        sho = sh @ my.mlp.sw2.t()
        print(f"  shared out finite={torch.isfinite(sho).all().item()}, "
              f"absmax={sho.abs().max().item():.3f}")
        # 单专家试算
        o = torch.zeros_like(x2)
        xs = x2[idx.reshape(-1)[0:1]] @ my.mlp.w1[idx[0][0]].t()
        print(f"  expert gemm finite={torch.isfinite(xs).all().item()}")
