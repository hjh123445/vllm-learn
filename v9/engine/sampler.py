# v5/engine/sampler.py
"""v4 采样器：torch.compile 用在这里。

为什么是这里而不是每层的算子：compile 的 guard 税按调用次数收（nano eager 的教训，
PERFORMANCE_NOTES §7.1）。采样器每步只调一次（每层算子每步调 28 次），
税可忽略；逐请求温度（greedy 行用 argmax 覆盖，修掉 nano 的 assert 禁 greedy）。"""
import os

import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        greedy = temperatures <= 1e-10
        safe = temperatures.clamp_min(1e-10).unsqueeze(1)
        probs = (logits.float() / safe).softmax(-1)
        tokens = torch.multinomial(probs, 1).squeeze(-1)
        return torch.where(greedy, logits.argmax(-1), tokens)


# TorchInductor compilation is useful for a long-lived serving process, but it
# launches a large compiler worker pool on the first sampling step.  That makes
# short distributed correctness/parallelism runs look hung and contaminates
# their timing.  Keep eager as the safe default and make compilation explicit.
if os.getenv("VLLM_LEARN_COMPILE_SAMPLER", "0") == "1":
    Sampler.forward = torch.compile(Sampler.forward, dynamic=True)
