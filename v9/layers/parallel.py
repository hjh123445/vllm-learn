# v9/layers/parallel.py
"""v9.3 并行组件架构：Linear 组件原生感知并行，weight_loader 直接装载本 rank 切片。

v6-v8 的做法（shard_model_for_tp：全量加载后就地切片）的三个结构性问题：
1. **每卡先持有全量权重再切**——Qwen3-30B 每卡瞬时 60GB；DSV4 的 149GB
   根本走不了这条路（被迫发明 layer_device_map）。组件架构从构造起就只有
   本 rank 的 1/N，装载时切片，GPU 上从不出现全量权重；
2. **切分逻辑在引擎侧伸手进模型内部**（`attn.q_proj.weight = ...`），
   模型文件对并行无感知。组件架构里并行语义归属组件，模型只管组装；
3. **绕过 weight_loader 协议**（先装载全量再切）——narrow().contiguous()
   的 view 陷阱（ENGINEERING_NOTES 2.0）正是"事后切"模式的产物。

组件清单：
  ParallelState         全局并行状态（tp/ep 坐标 + 通信组），launcher 在构造模型前注入
  ColumnParallelLinear  输出维切（q/k/v/w13）：本 rank 持 [N/tp, K]
  RowParallelLinear     输入维切（o/w2）：本 rank 持 [N, K/tp]，forward 带 allreduce
  shard_bounds          切分边界（支持头复制：kv_heads < tp 时多 rank 共享同一切片）

单卡零开销：tp=1 时组件退化为普通 Linear（weight [N,K] 全量，loader 直拷），
单卡/多卡是同一条代码路径——这是组件架构相对"事后切"的核心优势。
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


# ================= 全局并行状态 =================

class ParallelState:
    """进程的并行坐标 + 通信组。组件在 __init__ 时读取（构造即定型），
    launcher 必须在构造模型之前 configure。默认 tp=ep=cp=pp=1（单卡路径）。"""

    def __init__(self):
        self.tp_size, self.tp_rank = 1, 0
        self.ep_size, self.ep_rank = 1, 0
        self.cp_size, self.cp_rank = 1, 0       # DCP（decode cache 并行）
        self.pp_size, self.pp_rank = 1, 0       # PP（层切 stage）
        self.tp_group = None       # o_proj/w2 的 allreduce 域（tp 维）
        self.moe_group = None      # MoE 部分和的聚合域（tp×ep 张成的组）

    @property
    def tp_enabled(self):
        return self.tp_size > 1

    def __repr__(self):
        return (f"ParallelState(tp={self.tp_rank}/{self.tp_size}, "
                f"ep={self.ep_rank}/{self.ep_size})")


STATE = ParallelState()


def configure_parallel(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0,
                       tp_group=None, moe_group=None,
                       cp_size=1, cp_rank=0, pp_size=1, pp_rank=0):
    """launcher 注入（构造模型前调用）。通信组由引擎侧建好传入——
    组件不建组（v8.2 教训：new_group 必须全 rank 对称 + device_id + 预热）。"""
    STATE.tp_size, STATE.tp_rank = tp_size, tp_rank
    STATE.ep_size, STATE.ep_rank = ep_size, ep_rank
    STATE.cp_size, STATE.cp_rank = cp_size, cp_rank
    STATE.pp_size, STATE.pp_rank = pp_size, pp_rank
    STATE.tp_group, STATE.moe_group = tp_group, moe_group
    return STATE


def get_pp_indices(num_layers: int, pp_rank: int, pp_size: int):
    """PP 层区间（对齐 vLLM vllm/distributed/utils.get_pp_indices：
    均分 + 余数给前面的 stage）。返回 (start, end)。"""
    if pp_size == 1:
        return 0, num_layers
    base = num_layers // pp_size
    rem = num_layers % pp_size
    start = pp_rank * base + min(pp_rank, rem)
    end = start + base + (1 if pp_rank < rem else 0)
    return start, end


def tp_allreduce(x: torch.Tensor) -> torch.Tensor:
    """RowParallel/MoE 的部分和聚合（tp 组内）。单卡/未初始化直通。"""
    if STATE.tp_group is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, group=STATE.tp_group)
    return x


def moe_allreduce(x: torch.Tensor) -> torch.Tensor:
    """MoE 部分和聚合：partial 同时张成 tp（inter 切片）× ep（本地专家）两维，
    moe_group 是这两维张成的组。单卡直通。"""
    if STATE.moe_group is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, group=STATE.moe_group)
    return x


# ================= 切分数学 =================

def shard_bounds(out_dim: int, tp_size: int, tp_rank: int, min_out: int = 1):
    """输出维切分边界，返回 (start, length)。

    正常情形 out≥tp 且整除：均分（q_proj 4096/tp8=512）。
    头复制（out < tp 或 out/tp < min_out）：以 min_out 为最小单元切 n 份，
    rank r 取第 r*n//tp 份——kv_heads=4, tp=8, min_out=head_dim 时
    n=4，rank 0/1 共享 head0、rank 2/3 共享 head1……（GQA 对齐：
    q 头 j 映射 kv 头 j//(H/kv)，连续的 q 头块恰好落在共享的 kv 头上）。"""
    per = out_dim // tp_size
    if per < min_out:
        per = min_out
    n_shards = out_dim // per
    assert n_shards * per == out_dim, \
        f"out={out_dim} 切不动 per={per}（tp={tp_size}, min_out={min_out}）"
    shard = tp_rank * n_shards // tp_size
    return shard * per, per

# 注意：ColumnParallelLinear / RowParallelLinear 组件在 layers/linear.py
# （与 BaseLinear 同文件——它们是 Linear）。用模块级 __getattr__ 惰性
# re-export 保持旧 import 兼容（直接 import 会循环：linear → parallel → linear）
def __getattr__(name):
    if name in ("ColumnParallelLinear", "RowParallelLinear"):
        import layers.linear as _lin
        return getattr(_lin, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


