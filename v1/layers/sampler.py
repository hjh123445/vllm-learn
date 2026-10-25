import torch
import torch.nn as nn


class Sampler(nn.Module):
    def forward(self, logits: torch.Tensor, temperature: float = 0.0):
        """logits: [b, vocab] → token ids [b]"""
        if temperature < 1e-10:
            # greedy：argmax(logits) 等价于 argmax(softmax(logits))，先 softmax 是白算
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, 1, replacement=True).squeeze(-1)
