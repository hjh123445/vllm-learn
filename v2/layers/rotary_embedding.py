import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_emb(x,cos,sin):
    x1,x2 = torch.chunk(x.float(),2,-1)
    y1 = cos * x1 - sin * x2
    y2 = sin * x1 + cos * x2
    return torch.cat([y1,y2],dim = -1).to(x.dtype)


class RotaryEmbeddding(nn.Module):
    def __init__(self,
                 max_position,
                 head_dim,
                 base):
        super().__init__()
        exp = 1.0 / (base ** (torch.arange(0,head_dim,2,dtype=torch.float) / head_dim))
        seq = torch.arange(0,max_position,dtype=torch.float)
        
        use = torch.outer(seq,exp).float()
        cache = torch.cat([use.cos(),use.sin()],dim = -1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache",cache,persistent = False)
    def forward(self,positions:torch.Tensor,q:torch.Tensor,k:torch.Tensor):
        # cos_sin_cache: [max_pos, 1, head_dim] → 按 positions 索引得 [b, s, 1, hd]
        # 再 transpose(1,2) → [b, 1, s, hd]，才能广播到 q/k 的 [b, num_head, s, head_dim]
        cache = self.cos_sin_cache[positions].transpose(1, 2)
        cos,sin =torch.chunk(cache,2,dim = -1)
        q = apply_emb(q,cos,sin)
        k = apply_emb(k,cos,sin)
        return q,k




        
    