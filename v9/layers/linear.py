"""Linear 组件家族：BaseLinear（单卡）+ Column/RowParallelLinear（v9.3 并行）。

并行语义在组件内（构造读 ParallelState / weight_loader 吃全量内部切），
模型按普通 Linear 用，loader 对接 weight_loader 协议——三层职责分离。
切分数学与通信原语在 layers/parallel.py（非 Linear 的基础设施）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.parallel import STATE, shard_bounds, tp_allreduce


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
        return F.linear(x,self.weight,self.bias)
    
    def weight_loader(self, param, loaded, shard_id=None):
        assert param.shape == loader_weight.shape
        param.data.copy_(loader_weight)
    


# ============ v9.3 并行组件（与 BaseLinear 同文件：都是 Linear） ============

class ColumnParallelLinear(nn.Module):
    """输出维切分（Megatron 语义）：本 rank 持 [N/tp, K]，forward 无通信。
    用于 q_proj（按头切）/ k/v_proj（按头切，头不足时复制）/ w13（inter 切）。
    weight_loader 吃【全量】checkpoint 张量，切片装载——loader 无需感知并行。

    gate_up=True 时按 [gate(N/2) | up(N/2)] 两半各切再拼（本 rank 持
    [gate 的 I/tp | up 的 I/tp]）——适配 fc1/w13 的融合布局，
    使下游 chunk(2) 语义在 rank 内成立（直接均分 2I 会把 gate|up 劈开）。"""

    def __init__(self, in_dim: int, out_dim: int, min_out: int = 1, bias: bool = False,
                 gate_up: bool = False):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.gate_up = gate_up
        if gate_up:
            assert out_dim % 2 == 0
            half = out_dim // 2
            hs, hl = shard_bounds(half, STATE.tp_size, STATE.tp_rank)
            self.shard_start, self.shard_len = hs, hl          # 半区内的段
            self.weight = nn.Parameter(torch.empty(2 * hl, in_dim))
        else:
            start, length = shard_bounds(out_dim, STATE.tp_size, STATE.tp_rank, min_out)
            self.shard_start, self.shard_len = start, length
            self.weight = nn.Parameter(torch.empty(length, in_dim))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.weight.shape[0]))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param, loaded, shard_id=None):
        """loaded: 全量 [N, K] → 本 rank 切片。
        gate_up: [gate 的 I/tp | up 的 I/tp] 交错拼装。"""
        if self.gate_up:
            half = loaded.shape[0] // 2
            s, l = self.shard_start, self.shard_len
            w = torch.cat([loaded[s:s + l], loaded[half + s:half + s + l]], dim=0)
        else:
            w = loaded[self.shard_start:self.shard_start + self.shard_len]
        param.data.copy_(w.to(param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """输入维切分（Megatron 语义）：本 rank 持 [N, K/tp]。
    输入是本 rank 的列切片（来自上游 ColumnParallel），matmul 得部分和，
    forward 内做 tp allreduce——通信归属组件，模型层不再显式调用。
    用于 o_proj（q 头拼接维切）/ w2（inter 切）。"""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        assert in_dim % STATE.tp_size == 0, \
            f"in_dim={in_dim} 切不动 TP{STATE.tp_size}"
        self.shard_len = in_dim // STATE.tp_size
        self.shard_start = STATE.tp_rank * self.shard_len
        self.weight = nn.Parameter(torch.empty(out_dim, self.shard_len))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param, loaded, shard_id=None):
        """loaded: 全量 [N, K] → 列切片 [N, start:start+len]。
        ⚠ 必须 clone：narrow 视图共享旧 storage（v6 实测 61GB 纹丝不动的根因，
        ENGINEERING_NOTES 2.0）——这里 CPU→GPU 拷贝本身产生新存储，但
        copy_ 到 param 才是真正的归属转移。"""
        w = loaded[:, self.shard_start:self.shard_start + self.shard_len]
        param.data.copy_(w.to(param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        return tp_allreduce(out)


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