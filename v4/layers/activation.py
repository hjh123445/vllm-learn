import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
