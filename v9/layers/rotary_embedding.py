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
        self.max_position, self.head_dim, self.base = max_position, head_dim, base
        cache = self._build_cache(torch.device('cpu'), torch.float32)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self._cache_signature = (cache.device, cache.dtype)

    def _build_cache(self, device, dtype):
        # Match HF: initialize inverse frequencies in FP32 on CPU, then do
        # frequency multiplication and trig on the execution device in FP32.
        # Copying a CPU sin/cos table to CUDA is not numerically equivalent.
        inv = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2,
                    dtype=torch.float32) / self.head_dim))
        inv = inv.to(device=device)
        pos = torch.arange(self.max_position, device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=False):
            freq = (inv[None, :, None] @ pos[None, None, :]).transpose(1, 2)[0]
            cache = torch.cat((freq.cos(), freq.sin()), dim=-1)[:, None, :]
        return cache.to(dtype=dtype)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        cache = self.cos_sin_cache
        signature = (cache.device, cache.dtype)
        if signature != self._cache_signature:
            self.cos_sin_cache = self._build_cache(cache.device, cache.dtype)
            self._cache_signature = signature
        return self

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        """positions [total_q]；q [total_q, num_head, head_dim]；k 同。
        cos_sin_cache[positions] → [total_q, 1, head_dim]，头维=1 直接广播到所有头。"""
        cache = self.cos_sin_cache[positions]
        cos, sin = torch.chunk(cache, 2, dim=-1)
        return apply_emb(q, cos, sin), apply_emb(k, cos, sin)
