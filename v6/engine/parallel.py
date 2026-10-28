# v6/engine/parallel.py
"""v6 并行基础设施：进程组管理与 TP 通信原语。

设计原则（PARALLEL_ANALYSIS.md 的结论）：
  - TP 只在节点内（NVLink 域）；跨节点用 EP/PP/DP；
  - 通信原语薄封装：init_distributed（spawn 多进程）+ tp_allreduce（层输出同步）。

TP 切分约定（Megatron 风格）：
  - attention: q/k/v/o_proj 按头切（q 头数和 kv 头数都必须能整除 TP）——
    每卡算自己头的注意力，o_proj 后 allreduce；
  - MoE: w13 按行切（每卡的中间维度 = I/TP），w2 按列切——每卡算自己切片的
    expert 输出后 allreduce（router 全卡冗余）；
  - embedding/lm_head: vocab 切分（forward 时 all_gather 或 mask max）。
"""
import os

import torch
import torch.distributed as dist


def init_distributed(tp_size: int = 1, backend: str = "nccl"):
    """初始化进程组。单进程（tp_size=1）时不初始化——单卡路径零开销。
    env:// rendezvous 需要 RANK/WORLD_SIZE 三个变量（spawn 模式下由调用方设置）。"""
    if tp_size <= 1:
        return {"tp_size": 1, "tp_rank": 0, "world_size": 1, "rank": 0}
    os.environ.setdefault("WORLD_SIZE", str(tp_size))
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return {
        "tp_size": dist.get_world_size(),
        "tp_rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
    }


def tp_allreduce(x: torch.Tensor) -> torch.Tensor:
    """TP 的层输出同步。单卡直通（零开销）。"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    dist.all_reduce(x)
    return x


class ColumnParallelLinear:
    """w13 的行切（=输出的中间维度切）：每卡输出 [out/TP, in]。
    用 mixin 方式给现有 BaseLinear 加切分能力（避免改 layers/linear.py 的单卡路径）。"""

    @staticmethod
    def shard_weight(w: torch.Tensor, tp_rank: int, tp_size: int, dim: int = 0) -> torch.Tensor:
        """按 dim 切权重给本 rank。整除校验防静默错误。"""
        assert w.shape[dim] % tp_size == 0, \
            f"TP{tp_size} 切不动 dim{dim}={w.shape[dim]}（必须整除）"
        per = w.shape[dim] // tp_size
        return w.narrow(dim, tp_rank * per, per).contiguous()


def shard_model_for_tp(model, tp_rank: int, tp_size: int):
    """kv 头数 < TP 时（如 kv=4 < TP8）：kv 复制不切（每 rank 持有 ceil(kv/TP) 份的复制），
    q/o 仍切——attention 计算时多余 rank 重复算同一些 kv 头，正确性不受影响。"""
    """把已加载完整权重的模型就地切分到本 rank（内存换简单：先全量加载再切，
    生产实现应分片加载——本项目的验证优先）。
    切分点：attention 的 q/k/v/o（按头）、MoE 的 w13/w2（按中间维度）、
    embedding/lm_head（按 vocab，forward 用 allreduce 的 mask 方案则不切）。
    """
    h = model.config
    assert h.num_attention_heads % tp_size == 0
    per_q = h.num_attention_heads // tp_size
    # kv 头不够切时复制（TP8 kv4 → 每 rank 1 个 kv 头, 有 rank 共享同一切片）
    per_kv = max(1, h.num_key_value_heads // tp_size)
    kv_replicate = per_q // per_kv            # 本 rank 的 q 头组对应的 kv 复制组数
    per_inter = h.moe_intermediate_size // tp_size
    I_total = h.moe_intermediate_size

    # 切分前先释放全量权重的引用（narrow+contiguous 是新拷贝，旧 Parameter 随 gc 释放）
    for layer in model.model.layers:
        attn = layer.self_attn
        # q_proj [q_size, H] → 本 rank 的头 [per_q*hd, H]
        attn.q_proj.weight = __import__("torch").nn.Parameter(
            ColumnParallelLinear.shard_weight(attn.q_proj.weight.data, tp_rank, tp_size, 0),
            requires_grad=False)
        # kv 切分：tp_rank 映射到 kv 切片（TP>kv 时多个 rank 映射同一 kv 切片 = 复制）
        kv_slice = min(tp_rank, h.num_key_value_heads // per_kv - 1) if per_kv < h.num_key_value_heads \
            else tp_rank
        kv_shard_size = h.num_key_value_heads * attn.head_dim // max(1, h.num_key_value_heads // per_kv)
        n_kv_shards = max(1, h.num_key_value_heads // per_kv)
        kv_idx = tp_rank * n_kv_shards // tp_size
        attn.k_proj.weight = __import__("torch").nn.Parameter(
            attn.k_proj.weight.data.narrow(0, kv_idx * per_kv * attn.head_dim,
                                           per_kv * attn.head_dim).contiguous(),
            requires_grad=False)
        attn.v_proj.weight = __import__("torch").nn.Parameter(
            attn.v_proj.weight.data.narrow(0, kv_idx * per_kv * attn.head_dim,
                                           per_kv * attn.head_dim).contiguous(),
            requires_grad=False)
        # o_proj [H, q_size] 列切（输入是本 rank 的头）
        attn.o_proj.weight = __import__("torch").nn.Parameter(
            ColumnParallelLinear.shard_weight(attn.o_proj.weight.data, tp_rank, tp_size, 1),
            requires_grad=False)
        # 头数视角同步——外层 Qwen3MoeAttention 和内层 Attention 都要改
        # （triton kernel 的 GQA 映射读的是内层 Attention.num_kv_heads）
        attn.num_head = per_q
        attn.num_kv_head = per_kv
        attn.q_size = per_q * attn.head_dim
        attn.kv_size = per_kv * attn.head_dim
        if hasattr(attn, "attn"):
            attn.attn.num_heads = per_q
            attn.attn.num_kv_heads = per_kv
        # MoE: w13 布局是 [E, gate(I)|up(I), H] —— 每卡要拿 gate 的一半 + up 的
        # 一半（forward 里 chunk(2) 切 gate|up 后各自是 I/TP），不是按行整段切！
        moe = layer.mlp.experts
        w13 = moe.w13.data
        gate_r = w13[:, tp_rank * per_inter:(tp_rank + 1) * per_inter, :]      # [E, I/2, H]
        up_r = w13[:, I_total + tp_rank * per_inter:I_total + (tp_rank + 1) * per_inter, :]
        moe.w13 = __import__("torch").nn.Parameter(
            __import__("torch").cat([gate_r, up_r], dim=1).contiguous(), requires_grad=False)
        moe.w2 = __import__("torch").nn.Parameter(
            moe.w2.data.narrow(2, tp_rank * per_inter, per_inter).contiguous(),
            requires_grad=False)
        moe.inter = per_inter
        # router 全卡冗余（不切）——每卡都要算完整的 topk 路由
    torch.cuda.empty_cache()          # 释放被替换的全量权重
    return model
