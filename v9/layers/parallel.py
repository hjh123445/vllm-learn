"""Process-local parallel state, PP layer bounds and TP shard arithmetic.

Configure before constructing layers. EP coordinates derive from DP x TP;
parallel Linear implementations and weight loaders live in layers/linear.py.
"""

import os
import torch
import torch.distributed as dist


# ================= 全局并行状态 =================

class ParallelState:
    """进程的并行坐标 + 通信组。组件在 __init__ 时读取（构造即定型），
    launcher 必须在构造模型之前 configure。默认 tp=ep=cp=pp=1（单卡路径）。"""

    def __init__(self):
        self.tp_size, self.tp_rank = 1, 0
        self.dp_size, self.dp_rank = 1, 0
        self.replica_size, self.replica_rank = 1, 0  # diagnostic IDs, also for independent DP worlds
        self.enable_expert_parallel = False
        self.cp_size, self.cp_rank = 1, 0       # DCP（decode cache 并行）
        self.pp_size, self.pp_rank = 1, 0       # PP（层切 stage）
        self.pipeline_warmup = False
        self.pipeline_transport = None
        self.tp_group = None       # o_proj/w2 的 allreduce 域（tp 维）
        self.ep_group = None       # same PP/CP, union of DP x TP

    @property
    def ep_size(self):
        return self.dp_size * self.tp_size if self.enable_expert_parallel else 1

    @property
    def ep_rank(self):
        return self.dp_rank * self.tp_size + self.tp_rank if self.enable_expert_parallel else 0

    def __repr__(self):
        return (f"ParallelState(tp={self.tp_rank}/{self.tp_size}, "
                f"ep={self.ep_rank}/{self.ep_size})")


STATE = ParallelState()


def configure_parallel(tp_size=1, tp_rank=0,
                       tp_group=None,
                       ep_group=None,
                       cp_size=1, cp_rank=0, pp_size=1, pp_rank=0,
                       dp_size=1, dp_rank=0, enable_expert_parallel=False):
    """launcher 注入（构造模型前调用）。通信组由引擎侧建好传入——
    组件不建组（v8.2 教训：new_group 必须全 rank 对称 + device_id + 预热）。"""
    if any(not 0 <= rank < size for rank, size in (
            (dp_rank, dp_size), (tp_rank, tp_size),
            (pp_rank, pp_size), (cp_rank, cp_size))):
        raise ValueError("invalid DP/TP/PP/CP rank")
    STATE.tp_size, STATE.tp_rank = tp_size, tp_rank
    STATE.dp_size, STATE.dp_rank = dp_size, dp_rank
    STATE.replica_size = int(os.getenv("VLLM_LEARN_DP_REPLICA_COUNT", str(dp_size)))
    STATE.replica_rank = int(os.getenv("VLLM_LEARN_DP_REPLICA_ID", str(dp_rank)))
    STATE.enable_expert_parallel = enable_expert_parallel
    STATE.cp_size, STATE.cp_rank = cp_size, cp_rank
    STATE.pp_size, STATE.pp_rank = pp_size, pp_rank
    STATE.tp_group, STATE.ep_group = tp_group, ep_group
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
        from layers.batch_invariant import enabled, all_reduce
        if enabled():
            return all_reduce(x, STATE.tp_group)
        # Keep the numerical policy identical with/without graph capture.
        # FP32 is an explicit diagnostic/accuracy tradeoff, not a promise of
        # bitwise batch invariance (local BF16 GEMMs can also change with M).
        policy = os.getenv('VLLM_LEARN_TP_REDUCTION_DTYPE', 'model')
        if policy not in ('model', 'float32'):
            raise ValueError('TP reduction dtype must be model or float32')
        if policy == 'float32' and x.dtype in (torch.bfloat16, torch.float16):
            value = x.float()
            dist.all_reduce(value, group=STATE.tp_group)
            return value.to(x.dtype)
        dist.all_reduce(x, group=STATE.tp_group)
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
