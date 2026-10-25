import torch
import torch.nn as nn

import torch.nn.functional as F


class RmsNorm(nn.Module):
    def __init__(self,
                 hidden_dim,
                 eps):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_dim))
        self.eps = eps
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self,param,loader_weight):
        return param.data.copy_(loader_weight)

    def forward(self, x: torch.Tensor):
        # 全部 out-of-place：x.float() 在输入已是 fp32 时返回同一个张量，
        # 后续原地操作会改写调用方的 residual（fp32 下必错，bf16 下因拷贝而侥幸）
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # eps 此前漏加
        return (x * self.weight).to(dtype)        
    


