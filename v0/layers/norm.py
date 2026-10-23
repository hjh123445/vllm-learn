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

    def forward(self,x:torch.Tensor):
        dtype = x.dtype
        x = x.float()
        x1 = x.pow(2).mean(dim=-1,keepdim=True)
        x = x.mul_(torch.rsqrt(x1))
        return x.mul_(self.weight).to(dtype)        
    


