import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self,
                 vocab_dim,
                 hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_dim,hidden_dim))
        self.weight.weight_loader = self.weight_loader
    def weight_loader(self,param:torch.Tensor,loader_weight:torch.Tensor):
        param.data.copy_(loader_weight)
    def forward(self,x:torch.Tensor):
        return F.embedding(x,self.weight)
    

class LmHead(nn.Module):
    def __init__(self,
                 hidden_dim,
                 embedding_dim
                 ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dim,hidden_dim))
        self.weight.weight_loader = self.weight_loader
        
    def forward(self,x:torch.Tensor):
        return F.linear(x,self.weight)
    
    def weight_loader(self,param:torch.Tensor,loader_weight:torch.Tensor):
        param.data.copy_(loader_weight)