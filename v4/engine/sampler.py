# v4/engine/sampler.py
"""v4 采样器：torch.compile 用在这里。

为什么是这里而不是每层的算子：compile 的 guard 税按调用次数收（nano eager 的教训，
PERFORMANCE_NOTES §7.1）。采样器每步只调一次（每层算子每步调 28 次），
税可忽略；逐请求温度（greedy 行用 argmax 覆盖，修掉 nano 的 assert 禁 greedy）。"""
import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile(dynamic=True)
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        greedy = temperatures <= 1e-10
        safe = temperatures.clamp_min(1e-10).unsqueeze(1)
        probs = (logits.float() / safe).softmax(-1)
        tokens = torch.multinomial(probs, 1).squeeze(-1)
        return torch.where(greedy, logits.argmax(-1), tokens)
