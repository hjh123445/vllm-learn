import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLinear(nn.Module):
    def __init__(self,in_dim,out_dim,bias = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim,in_dim))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
    
    def forward(self,x:torch.Tensor):
        # dtype 自适应：上游 fp32 中间量（rope/norm 的 fp32 计算）不再炸下游
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        return F.linear(x,self.weight,self.bias)
    
    def weight_loader(self,param:torch.Tensor,loader_weight:torch.Tensor):
        assert param.shape == loader_weight.shape
        param.data.copy_(loader_weight)
    


if __name__ == "__main__":
    x = torch.randn(2,1024,device = "cuda:0",dtype = torch.float16)
    weight = torch.randn(4096,1024,device = "cuda:0",dtype = torch.float16)
    net = BaseLinear(1024,4096).cuda().to(torch.float16)
    loader = net.weight_loader
    loader(net.weight,weight)
    y = net(x)
    net1 = nn.Linear(1024,4096,bias=False).cuda(0).to(torch.float16)
    loader(net1.weight,weight)
    y1 = net1(x)
    print(torch.allclose(y, y1, rtol=1e-05, atol=1e-08))