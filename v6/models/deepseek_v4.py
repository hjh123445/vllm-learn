# v6/models/deepseek_v4.py
"""DeepSeek-V4-Flash 模型（与 qwen3.py / qwen3_moe.py 并列，互不排斥）。

架构（43 层，每层组件）：
  MLA 注意力（Shared-KV MQA + partial RoPE + sink + 分组输出投影 o_a/o_b）
  + MoE（前 3 层 hash 路由 tid2eid 查表，其余 sqrtsoftplus 路由；
    256 专家 top-6 + 1 shared；FP8 block 量化权重）
  + HyperConnections（4 残差流贯穿全模型，替代普通残差）
  + HCA/CSA 压缩 KV（长上下文）

权重的 FP8 反量化在 loader 侧做（int8 存储 + e8m0 scale）——模型参数
直接持有反量化后的 bf16。

v6.3 阶段三第一版（正确性优先）：
  - HCA/CSA 压缩器暂不接入（无 cache 的单发模式）——短序列影响小
  - yarn rope 用标准 RoPE 近似（theta=10000，factor=16 的 yarn 缩放留后）
  - 上述两点记录在 MILESTONES 的 known-simplifications
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.linear import BaseLinear
from layers.norm import RmsNorm
from layers.dsv4_components import (
    HyperConnection, HCACompressor, mla_attention, apply_partial_rope)


# ---------- 平铺版 RoPE（theta 可配，yarn 近似留后） ----------

class DSV4Rope(nn.Module):
    """yarn 缩放的 RoPE（V4-Flash: factor=16, original_max=65536, theta=10000）。
    yarn 的低频/高频斜坡基于 beta_fast/beta_slow，这里用标准实现。"""

    def __init__(self, head_dim, rope_dim, base=10000, max_position=8192,
                 yarn_factor=16.0, yarn_original_max=65536,
                 beta_fast=32.0, beta_slow=1.0):
        super().__init__()
        self.rope_dim = rope_dim
        # pos / factor 作为有效位置（长序列外推压缩）
        low, high = yarn_original_max // yarn_factor, yarn_original_max
        inv = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim))
        # yarn 斜坡：低频维度缩 factor，高频不动
        if isinstance(yarn_factor, float) and yarn_factor > 1.0 and low < high:
            import math
            floor_f, ceil_f = math.floor(rope_dim / 2 * math.log(high / low, base)), \
                math.ceil(rope_dim / 2 * math.log(high / low, base))
            bl, bh = beta_fast, beta_slow
            ramp = torch.zeros(rope_dim // 2)
            lo_idx, hi_idx = max(0, (rope_dim // 2) - ceil_f) - bl, \
                min(rope_dim // 2, floor_f) + bh
            idx = torch.arange(rope_dim // 2, dtype=torch.float)
            ramp = torch.clamp((idx - lo_idx) / max(hi_idx - lo_idx, 1), 0.0, 1.0)
            inv = inv / (1 + (yarn_factor - 1) * ramp)   # 高频(ramp=1)缩factor, 低频不动
            self._attention_scaling = 0.1 * math.log(yarn_factor) + 1.0
        else:
            self._attention_scaling = 1.0
        pos = torch.arange(max_position, dtype=torch.float) / (yarn_factor if yarn_factor > 1 else 1.0)
        freq = torch.outer(pos, inv)
        self.register_buffer("cos", freq.cos() * self._attention_scaling, persistent=False)
        self.register_buffer("sin", freq.sin() * self._attention_scaling, persistent=False)

    def get(self, positions):
        """positions [T] → (cos [T, rd/2], sin)。rope buffer 固定在一张卡
        （共享注入），跨卡层取用时搬运。"""
        dev = positions.device
        buf_dev = self.cos.device
        if dev != buf_dev:
            idx = positions.to(buf_dev)
            return self.cos[idx].to(dev), self.sin[idx].to(dev)
        return self.cos[positions], self.sin[positions]


# ---------- MLA 注意力 ----------

class DSV4Attention(nn.Module):
    """Shared-KV MQA + partial RoPE + sink + 分组输出投影。平铺 [T, H] 输入。"""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.H = cfg["num_attention_heads"]
        self.D = cfg["head_dim"]
        self.rd = cfg["qk_rope_head_dim"]          # rope 维（partial）
        self.scale = self.D ** -0.5
        self.rope = None                            # 由模型注入共享

        self.q_a_proj = BaseLinear(cfg["hidden_size"], cfg["q_lora_rank"])
        self.q_a_norm = RmsNorm(cfg["q_lora_rank"], eps=cfg["rms_norm_eps"])
        self.q_b_proj = BaseLinear(cfg["q_lora_rank"], self.H * self.D)
        self.kv_proj = BaseLinear(cfg["hidden_size"], self.D)
        self.kv_norm = RmsNorm(self.D, eps=cfg["rms_norm_eps"])
        # 分组输出投影：64 头×512 = 32768 维直接投到 4096 太贵——
        # 8 组各自 4096→1024（o_a），拼 8192 → 4096（o_b）
        g = cfg["o_groups"]
        self.o_groups = g
        self.o_lora = cfg["o_lora_rank"]
        self.o_a = BaseLinear(self.H * self.D // g, g * self.o_lora)   # 块对角用法
        self.o_b = BaseLinear(g * self.o_lora, cfg["hidden_size"])
        self.sinks = nn.Parameter(torch.empty(self.H))
        self.sinks.weight_loader = self._load_sink

    def _load_sink(self, param, loaded, shard_id=None):
        param.data.copy_(loaded)

    @staticmethod
    def _q_b_norm(x):
        # HF DeepseekV4UnweightedRMSNorm（无学习权重——与带权重的 q_a_norm 不同）
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)

    def forward(self, x, positions, attn_mask=None):
        """x [T, hidden]; positions [T] → [T, hidden]。"""
        T = x.shape[0]
        cos, sin = self.rope.get(positions)

        qr = self.q_a_norm(self.q_a_proj(x))                       # [T, q_lora]
        q = self.q_b_proj(qr).view(T, self.H, self.D)
        q = self._q_b_norm(q)                                     # 无权重 RMSNorm
        q = apply_partial_rope(q, cos, sin, self.rd)
        kv = self.kv_norm(self.kv_proj(x))                         # [T, D] 单 KV 头
        kv = apply_partial_rope(kv.unsqueeze(1), cos, sin, self.rd).squeeze(1)

        out = mla_attention(q, kv, self.sinks, self.scale, attn_mask, cos, sin)

        # 分组输出投影（块对角）：[T, H*D] → 8 组 [T, 4096] → 各投 1024
        grouped = out.reshape(T, self.o_groups, -1)                 # [T, g, H*D/g]
        # o_a 是 [g*lora, H*D/g] 的整权重——按组取块对角
        w = self.o_a.weight.view(self.o_groups, self.o_lora, -1)    # [g, lora, in]
        a = torch.einsum("tgi,gli->tgl", grouped.float(), w.float()).to(grouped.dtype)
        return self.o_b(a.reshape(T, -1))


# ---------- MoE ----------

class DSV4MoE(nn.Module):
    """256 专家 top-6 + 1 shared。hash 路由（前 3 层）或 sqrtsoftplus。"""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        E = cfg["n_routed_experts"]
        I = cfg["moe_intermediate_size"]
        H = cfg["hidden_size"]
        self.E, self.I, self.H = E, I, H
        self.k = cfg["num_experts_per_tok"]
        self.hash = layer_idx < cfg["num_hash_layers"]
        self.scaling = cfg["routed_scaling_factor"]

        self.gate_weight = nn.Parameter(torch.empty(E, H))
        self.gate_weight.weight_loader = self._load_gate
        self.gate_bias = nn.Parameter(torch.zeros(E))     # e_score_correction_bias
        self.gate_bias.weight_loader = self._load_gate
        if self.hash:
            self.register_buffer("tid2eid", torch.zeros(cfg["vocab_size"], self.k, dtype=torch.long))

        # 专家权重（FP8 反量化后装填）
        self.w1 = nn.Parameter(torch.empty(E, I, H))     # gate
        self.w3 = nn.Parameter(torch.empty(E, I, H))     # up
        self.w2 = nn.Parameter(torch.empty(E, H, I))     # down
        # shared expert（单份，不走路由）
        self.sw1 = nn.Parameter(torch.empty(I, H))
        self.sw3 = nn.Parameter(torch.empty(I, H))
        self.sw2 = nn.Parameter(torch.empty(H, I))

    def _load_gate(self, param, loaded, shard_id=None):
        param.data.copy_(loaded)

    def weight_loader(self, param, loaded, shard_id):
        """shard_id: ('expert', i) / ('shared', None)——FP8 反量化后装填。"""
        kind, i = shard_id
        if kind == "expert":
            if param is self.w1:  param.data[i] = loaded
            elif param is self.w3: param.data[i] = loaded
            else:                  param.data[i] = loaded
        else:
            if param is self.sw1: param.data.copy_(loaded)
            elif param is self.sw3: param.data.copy_(loaded)
            else: param.data.copy_(loaded)

    def _route(self, x, input_ids):
        dev = x.device
        logits = F.linear(x.float(), self.gate_weight.float())
        scores = F.softplus(logits).sqrt()                          # sqrtsoftplus
        if self.hash:
            idx = self.tid2eid[input_ids.reshape(-1)].long()        # [T, k] 静态查表
        else:
            idx = (scores + self.gate_bias.float()).topk(self.k, dim=-1).indices
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return idx.to(dev), (w * self.scaling).to(dev)

    def forward(self, x, input_ids):
        T, H = x.shape
        idx, w = self._route(x, input_ids)

        out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
        flat_e = idx.reshape(-1)
        flat_t = torch.arange(T, device=x.device).repeat_interleave(self.k)
        flat_w = w.reshape(-1)
        counts = torch.bincount(flat_e, minlength=self.E)
        offs = counts.cumsum(0)
        order = torch.argsort(flat_e, stable=True)
        s_t, s_w = flat_t[order], flat_w[order]
        start = 0
        for e, end in enumerate(offs.tolist()):
            if end == start:
                continue
            toks = s_t[start:end]
            h1 = x[toks] @ self.w1[e].t()
            h3 = x[toks] @ self.w3[e].t()
            act = F.silu(h1) * h3
            o = act @ self.w2[e].t()
            out.index_add_(0, toks, o * s_w[start:end, None].to(o.dtype))
            start = end
        # shared expert（全 token）
        sh = F.silu(x @ self.sw1.t()) * (x @ self.sw3.t())
        return out + sh @ self.sw2.t()


# ---------- Decoder 层（HC 残差流） ----------

class DSV4DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.self_attn = DSV4Attention(cfg, layer_idx)
        self.mlp = DSV4MoE(cfg, layer_idx)
        self.input_layernorm = RmsNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = RmsNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        hc = cfg.get("hc_mult", 4)
        self.attn_hc = HyperConnection(hc, cfg["hidden_size"], cfg.get("hc_sinkhorn_iters", 20))
        self.ffn_hc = HyperConnection(hc, cfg["hidden_size"], cfg.get("hc_sinkhorn_iters", 20))

    def forward(self, streams, positions, input_ids):
        """streams [T, hc, H] → [T, hc, H]。HC 残差流贯穿。"""
        dtype = streams.dtype
        post, comb, collapsed = self.attn_hc(streams)
        attn_out = self.self_attn(self.input_layernorm(collapsed), positions)
        streams = (post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2)
                   + torch.matmul(comb.to(dtype).transpose(-1, -2), streams))

        post, comb, collapsed = self.ffn_hc(streams)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed), input_ids)
        return (post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2)
                + torch.matmul(comb.to(dtype).transpose(-1, -2), streams))


# ---------- 模型 ----------

class DSV4Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(
            [DSV4DecoderLayer(cfg, i) for i in range(cfg["num_hidden_layers"])])
        self.norm = RmsNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.hc_head = None       # 由 DSV4ForCausalLM 注入
        rope = DSV4Rope(cfg["head_dim"], cfg["qk_rope_head_dim"],
                        base=cfg.get("rope_theta", 10000))
        for l in self.layers:
            l.self_attn.rope = rope

    def forward(self, input_ids, positions):
        """平铺 [T] 输入。初始流：stream 0 = embed，其余 0。
        层可能分放在不同卡（DSV4 反量化后 150GB）——层间自动搬运 streams。"""
        T = input_ids.shape[0]
        hc = self.cfg.get("hc_mult", 4)
        # embed 可能在远端卡（分层分放）——输入先搬到 embed 所在卡
        emb_dev = self.embed.weight.device
        input_ids = input_ids.to(emb_dev)
        positions = positions.to(emb_dev)
        streams = torch.zeros(T, hc, self.cfg["hidden_size"],
                              dtype=self.embed.weight.dtype, device=emb_dev)
        streams[:, 0] = self.embed(input_ids)
        for layer in self.layers:
            dev = layer.input_layernorm.weight.device       # 层的归属卡
            if streams.device != dev:
                streams = streams.to(dev)                   # 跨卡搬运激活（MB 级）
            streams = layer(streams, positions.to(dev), input_ids.to(dev))
        out_dev = self.norm.weight.device
        if streams.device != out_dev:
            streams = streams.to(out_dev)
        return self.norm(self.hc_head(streams))


class DSV4ForCausalLM(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        with open(os.path.join(model_dir, "config.json")) as f:
            raw = json.load(f)
        cfg = {k: raw[k] for k in [
            "hidden_size", "num_hidden_layers", "num_attention_heads", "head_dim",
            "qk_rope_head_dim", "q_lora_rank", "o_groups", "o_lora_rank",
            "vocab_size", "rms_norm_eps", "rope_theta", "n_routed_experts",
            "n_shared_experts", "moe_intermediate_size", "num_experts_per_tok",
            "num_hash_layers", "routed_scaling_factor", "hc_mult",
            "hc_sinkhorn_iters", "max_position_embeddings"]}
        self.cfg = cfg
        self.model = DSV4Model(cfg)
        from layers.dsv4_components import HyperHead
        self.model.hc_head = HyperHead(cfg.get("hc_mult", 4), cfg["hidden_size"])
        self.lm_head = BaseLinear(cfg["hidden_size"], cfg["vocab_size"])

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, x):
        return self.lm_head(x)
