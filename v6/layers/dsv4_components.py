# v6/layers/dsv4_components.py
"""v6.3 阶段二：DeepSeek-V4 的独有组件单元实现（与 qwen3 系组件并列，互不排斥）。

三个组件（全部纯 PyTorch 数学，对拍 HF 5.12 源码）：
1. MLA 注意力（Shared-KV MQA + partial RoPE + attention sink + 共轭反旋转）
2. HyperConnection（4 残差流 + Sinkhorn 双随机混合）
3. HCA 压缩器（128:1 窗口压缩，长上下文的核心）

设计要点（对我们平铺批架构的适配在模型文件里做，这里是数学核心）：
- K=V 共享：kv_proj 出 [T, head_dim] 单头，广播到 64 个 q 头；
- partial RoPE：只有每头前 qk_rope_head_dim(64)/512 维做 rope；
- 输出共轭反旋转：V 带 rope，输出在投影前用 -sin 抵消（相对位置的关键）；
- sink：softmax 时 cat 一个可学习标量，防长序列稀释。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ 1. MLA 注意力核心数学 ============

def mla_attention(q, kv, sink, scale, mask=None, cos=None, sin=None):
    """DeepSeek-V4 MLA 的 eager 数学。
    q:  [T, H, D]（已 rope）
    kv: [L, D]   单 KV 头（已 rope，K=V 共享）
    sink: [H]    每头可学习 sink logit
    cos/sin: 输出反旋转用（-sin 共轭）
    返回 [T, H, D]（已反旋转，可直接 o_proj）。"""
    T, H, D = q.shape
    L = kv.shape[0]
    in_dtype = q.dtype                          # 出口转回（下游线性层是 bf16）
    # fp32 计算（bf16 下 einsum 要求同 dtype 且 softmax 精度不够）
    q, kv = q.float(), kv.float()
    # scores: 每头与单 KV 头的点积 → [H, T, L]
    scores = torch.einsum("thd,ld->htl", q, kv) * scale
    # causal（含压缩条目的全局位置——条目 w 的全局位置由调用方保证 ≤ 因果界时
    # 已经等价；这里用 kv_len 前缀=全量 KV, 压缩条目用其窗口右端位置）：
    # q 位置 t 可见 kv 位置 l ≤ t。mask=None 时若 L > T 也不加（调用方自管）。
    if mask is not None:
        scores = scores + mask.float()
    # attention sink：cat 到 key 维尾部参与 softmax（gpt-oss 式）
    sink_logits = sink.view(H, 1, 1).expand(H, T, 1)
    combined = torch.cat([scores, sink_logits], dim=-1)
    probs = combined.softmax(dim=-1)
    probs = probs[..., :-1]                     # drop sink（它只吸收概率不贡献值）
    out = torch.einsum("htl,ld->thd", probs, kv)  # V=kv
    # 共轭反旋转（对齐 HF：attention 后用 -sin 再 apply 一次把 V 的 rope 抵消）
    if cos is not None and sin is not None:
        rd = cos.shape[-1] * 2
        neg_sin = -sin
        out = apply_partial_rope(out, cos, neg_sin, rd)
    return out.to(in_dtype)


def apply_partial_rope(x, cos, sin, rope_dim):
    """V4 的 interleaved partial RoPE（对齐 HF apply_rotary_pos_emb）。

    三个关键点：
    1. rope 维在【尾部】x[..., -rope_dim:]，nope 在头部（V4 布局）；
    2. 偶奇交错配对（rotate_half: x[0::2] 与 x[1::2]），
       cos/sin 是半宽（rd/2 个频率），repeat_interleave(2) 展开；
    3. fp32 计算后转回原 dtype。
    x: [T, H, D]（或 [T, D]）; cos/sin: [T, rope_dim//2]（fp32）。"""
    dtype = x.dtype
    ndim_extra = x.dim() - 2               # [T, (H,) D] → unsqueeze 到头维后 1
    c = cos.repeat_interleave(2, dim=-1)    # [T, rd]
    s = sin.repeat_interleave(2, dim=-1)
    if x.dim() == 3:                        # [T, H, D]
        c, s = c.unsqueeze(1), s.unsqueeze(1)
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    x1 = rope[..., 0::2].float()
    x2 = rope[..., 1::2].float()
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)  # rotate_half
    rotated = (rope.float() * c) + (rot * s)
    return torch.cat([nope, rotated.to(dtype)], dim=-1)


# ============ 2. HyperConnection（mHC） ============

class HyperConnection(nn.Module):
    """4 残差流的混合。weight_loader 兼容（hc_ffn_fn/base/scale 直装）。
    数学（HF §DeepseekV4HyperConnection）:
      pre  = σ(fln(streams)·scale₀ + base₀) + eps     流坍缩权重（进子层）
      post = 2σ(fln·scale₁ + base₁)                   子层输出放置权重
      comb = Sinkhorn(softmax(fln·scale₂ + base₂))     流混合矩阵（双随机）"""

    def __init__(self, hc_mult=4, hidden=4096, sinkhorn_iters=20, eps=1e-6):
        super().__init__()
        self.hc, self.eps, self.iters = hc_mult, eps, sinkhorn_iters
        mix = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.empty(mix, hc_mult * hidden))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))
        self.hidden = hidden

    def weight_loader(self, param, loaded, shard_id=None):
        param.data.copy_(loaded)

    def _unweighted_rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def forward(self, streams):
        """streams: [T, hc, hidden] → (post [T, hc], comb [T, hc, hc], collapsed [T, hidden])。"""
        hc = self.hc
        T = streams.shape[0]
        flat = self._unweighted_rms(streams.reshape(T, hc * self.hidden).float())
        logits = F.linear(flat, self.fn.float())
        pre_w, post_w, comb_w = logits.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        s0, s1, s2 = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * s0 + pre_b[None]) + self.eps       # [T, hc]
        post = 2 * torch.sigmoid(post_w * s1 + post_b[None])           # [T, hc]
        comb = torch.softmax((comb_w.view(T, hc, hc) * s2
                              + comb_b.view(hc, hc)[None]), dim=-1) + self.eps
        # Sinkhorn 投影到双随机流形
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)

        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=1).to(streams.dtype)
        return post, comb, collapsed


class HyperHead(nn.Module):
    """末端流坍缩（最终 norm 前）。"""

    def __init__(self, hc_mult=4, hidden=4096, eps=1e-6):
        super().__init__()
        self.hc, self.eps = hc_mult, eps
        self.hidden = hidden
        self.hc_fn = nn.Parameter(torch.empty(hc_mult, hc_mult * hidden))
        self.hc_base = nn.Parameter(torch.empty(hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def weight_loader(self, param, loaded, shard_id=None):
        param.data.copy_(loaded)

    def _unweighted_rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def forward(self, streams):
        T = streams.shape[0]
        flat = self._unweighted_rms(streams.reshape(T, self.hc * self.hidden).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()[None]) + self.eps
        return (pre.unsqueeze(-1) * streams).sum(dim=1).to(streams.dtype)


# ============ 3. HCA 压缩器（128:1 窗口压缩） ============

class HCACompressor(nn.Module):
    """每 128 个源 token 压成 1 个 KV 条目（长上下文核心）。
    数学（HF §DeepseekV4HCACompressor eqs.20-23）:
      C_i = Σ_{j∈window} softmax(gate_j + pos_bias)_j ⊙ kv_j   → kv_norm → rope
    """

    def __init__(self, hidden=4096, head_dim=512, compress_rate=128, eps=1e-6):
        super().__init__()
        self.rate = compress_rate
        self.head_dim = head_dim
        self.kv_proj = nn.Linear(hidden, head_dim, bias=False)
        self.gate_proj = nn.Linear(hidden, head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(compress_rate, head_dim))
        self.norm = nn.RMSNorm(head_dim, eps=eps)

    def weight_loader(self, param, loaded, shard_id=None):
        param.data.copy_(loaded)

    def forward(self, hidden_states, apply_rope=None):
        """hidden: [T, H] → compressed [T/128, head_dim]。
        apply_rope: 可调用（compressed, positions）→ rope 后的（无则跳过 rope）。"""
        T = hidden_states.shape[0]
        usable = (T // self.rate) * self.rate
        if usable == 0:
            return hidden_states.new_zeros(0, self.head_dim)
        kv = self.kv_proj(hidden_states[:usable])                     # [U, D]
        gate = self.gate_proj(hidden_states[:usable])                 # [U, D]
        n_win = usable // self.rate
        kv = kv.view(n_win, self.rate, self.head_dim)
        gate = gate.view(n_win, self.rate, self.head_dim) + self.position_bias[None]
        weights = gate.softmax(dim=1, dtype=torch.float32).to(kv.dtype)
        compressed = self.norm((kv * weights).sum(dim=1))             # [n_win, D]
        if apply_rope is not None:
            positions = torch.arange(n_win, device=kv.device) * self.rate
            compressed = apply_rope(compressed, positions)
        return compressed


# ============ v6.3 数值修复：CSA 压缩器 + Lightning Indexer（对齐 HF） ============

class CSACompressor(nn.Module):
    """Compressed Sparse Attention 压缩器（21/43 层, compress_rate=4）。

    数学（HF DeepseekV4CSACompressor, paper §2.3.1）：
      1. kv/gate 投影到 2*head_dim：Ca（[.., :D] 贡献给下一窗口）与
         Cb（[.., D:] 贡献给当前窗口）两个独立压缩序列；
      2. 窗口 w 的压缩条目 = softmax(gate + pos_bias) 加权的
         [上一窗的 Ca | 当前窗的 Cb] 拼接（宽 2m, 步 m）——相邻窗口共享一半
         槽位（overlap）,首窗的 Ca 半槽为零 kv/-inf gate（softmax 权重 0）;
      3. kv_norm → 在位置 w*m 处 rope（compress theta=160000）;
      4. Lightning Indexer 从 n_win 个条目里为每个 query 选 top-k,
         其余条目 block_bias=-inf（index_topk=512, 短序列 topk=全部 →
         退化为因果掩码——e2e 长度下的真实影响主要是压缩条目本身）。
    平铺 [T, H] 输入（与 HCACompressor 同风格, cache-less 全量重算模式）。"""

    def __init__(self, hidden=4096, head_dim=512, compress_rate=4, eps=1e-6):
        super().__init__()
        self.rate = compress_rate
        self.D = head_dim
        self.kv_proj = nn.Linear(hidden, 2 * head_dim, bias=False)
        self.gate_proj = nn.Linear(hidden, 2 * head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(compress_rate, 2 * head_dim))
        self.norm = nn.RMSNorm(head_dim, eps=eps)

    def weight_loader(self, param, loaded, shard_id=None):
        param.data.copy_(loaded)

    def forward(self, hidden, q_residual, positions, apply_rope=None):
        """hidden [T, H] → compressed_kv [n_win, D]（已 rope）。"""
        T = hidden.shape[0]
        usable = (T // self.rate) * self.rate
        if usable == 0:
            return hidden.new_zeros(0, self.D)
        kv = self.kv_proj(hidden[:usable]).view(-1, self.rate, 2 * self.D)
        gate = self.gate_proj(hidden[:usable]).view(-1, self.rate, 2 * self.D)
        gate = gate + self.position_bias[None]
        n_win = kv.shape[0]
        # overlap 布局 [n_win, 2m, D]: 后半 = 本窗 Cb, 前半 = 上窗 Ca
        # （首窗前半零 kv/-inf gate → softmax 权重 0, 数学上自然失效）
        m = self.rate
        new_kv = torch.zeros(n_win, 2 * m, self.D, dtype=kv.dtype, device=kv.device)
        new_gate = torch.full((n_win, 2 * m, self.D), float("-inf"),
                              dtype=gate.dtype, device=gate.device)
        new_kv[:, m:] = kv[..., self.D:]
        new_gate[:, m:] = gate[..., self.D:]
        if n_win > 1:
            new_kv[:, 1:, :m] = kv[:, :-1, :, :self.D]
            new_gate[:, 1:, :m] = gate[:, :-1, :, :self.D]
        w = new_gate.softmax(dim=1, dtype=torch.float32).to(kv.dtype)
        compressed = self.norm((new_kv * w).sum(dim=1))              # [n_win, D]
        if apply_rope is not None:
            cpos = torch.arange(n_win, device=hidden.device) * m
            compressed = apply_rope(compressed, cpos)
        return compressed
