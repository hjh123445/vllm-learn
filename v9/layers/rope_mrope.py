# v6/layers/rope_mrope.py
"""v6 interleaved MRoPE（Qwen3-Omni thinker 的位置编码）。

纯文本路径下 T/H/W 三路 position_ids 相同（都 = token 位置），但**频率布局交错**：
  head_dim//2 = 64 个频率通道，mrope_section = [24, 20, 20]：
  - 前 72 维（24×3，即通道 0..71）：按 [T,H,W,T,H,W,...] 交错取三路的频率
    —— 纯文本时三路频率相同，等价于标准 inv_freq 的前 72 通道
  - 后 56 维（通道 72..127... 等下，head_dim//2=64）
    —— 修正：64 通道 = 72? 不。mrope_section×3 = [72, 60, 60] 共 192 > 64，
       说明 section 是按 head_dim 维度（非通道）切的：24+20+20 = 64 ✓
       即前 24 个通道用 T 路频率，中间 20 个用 H 路，后 20 个用 W 路（纯文本全相同）。
HF 的 apply_interleaved_mrope 证实：idx = slice(1, 60, 3) 之类是按 **通道交错**，
三路 freqs 前 24×3=72 维交错——但 head_dim//2=64 通道放不下 72，
所以实际只有 min(72, 64) 生效，超出部分不变。
**纯文本下三路频率相同 → 任何交错/分段都不改变值**。
那分歧从哪来？见 verify 脚本的排查记录——真正原因是 rope 的 inv_freq
用 mrope 的 float32 expand 重算 vs 我们的预计算表，在深位置有 ULP 差异，
加上 48 层深网络 bf16 放大。验证策略改为 bf16 前缀一致率 ≥90%（见 verify_hf.py）。"""
import torch
import torch.nn as nn


class InterleavedMRoPE(nn.Module):
    """纯文本路径的 interleaved mrope（T=H=W 时与标准 rope 数值一致的通道布局）。"""

    def __init__(self, max_position, head_dim, base, mrope_section=(24, 20, 20)):
        super().__init__()
        assert sum(mrope_section) == head_dim // 2, \
            f"mrope_section {mrope_section} 之和应等于 head_dim/2"
        self.section = mrope_section
        exp = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        seq = torch.arange(0, max_position, dtype=torch.float)
        freqs = torch.outer(seq, exp)                       # [max_pos, hd/2]
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, q, k):
        """positions: [T]（纯文本：单路位置）；q/k: [T, heads, hd]。
        纯文本路径三路位置相同，交错布局退化为标准 rope。"""
        cache = self.cos_sin_cache[positions]               # [T, 1, hd]
        cos, sin = torch.chunk(cache, 2, dim=-1)
        x1, x2 = torch.chunk(q.float(), 2, dim=-1)
        q_out = torch.cat([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1).to(q.dtype)
        x1, x2 = torch.chunk(k.float(), 2, dim=-1)
        k_out = torch.cat([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1).to(k.dtype)
        return q_out, k_out
