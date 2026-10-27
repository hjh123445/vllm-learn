import torch
import torch.nn as nn


def apply_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, -1)
    y1 = cos * x1 - sin * x2
    y2 = sin * x1 + cos * x2
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


class RotaryEmbeddding(nn.Module):
    """v1 混合批次版：positions 是平铺的 [total_q]（flattened batch，无 b/s 维）。"""

    def __init__(self, max_position, head_dim, base):
        super().__init__()
        exp = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        seq = torch.arange(0, max_position, dtype=torch.float)
        use = torch.outer(seq, exp).float()
        cache = torch.cat([use.cos(), use.sin()], dim=-1).unsqueeze_(1)   # [max_pos, 1, head_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        """positions [total_q]；q [total_q, num_head, head_dim]；k 同。
        cos_sin_cache[positions] → [total_q, 1, head_dim]，头维=1 直接广播到所有头。"""
        cache = self.cos_sin_cache[positions]
        cos, sin = torch.chunk(cache, 2, dim=-1)
        return apply_emb(q, cos, sin), apply_emb(k, cos, sin)
