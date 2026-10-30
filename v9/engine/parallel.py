# v6/engine/parallel.py（v9 中已被 layers/parallel.py 的组件架构取代）
"""⚠ DEPRECATED（v9.3 起）：本文件的 shard_model_for_tp/ep 是 v6 的"全量加载
后就地切分"模式——每卡瞬时持有全量权重、切分逻辑伸手进模型内部、绕过
weight_loader 协议。v9 的并行组件架构（layers/parallel.py：ColumnParallel/
RowParallelLinear + GroupedExperts 内置 EP/TP）从构造起只持本 rank 权重。
保留此文件仅为 v6-v8 历史快照的引用完整性；新代码勿用。

v6 并行基础设施：进程组管理与 TP 通信原语。

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


_TP_GROUP = None    # 由 set_tp_group 设置（TP×EP 组合时 ≠ world group）
_TP_ENABLED = False # 由 set_tp_enabled 设置——PP 等模式必须禁用
                     # （tp_allreduce 只看 world_size 会误伤）


def set_tp_enabled(enabled: bool):
    global _TP_ENABLED
    _TP_ENABLED = enabled


def set_tp_group(group):
    global _TP_GROUP
    _TP_GROUP = group


def tp_allreduce(x: torch.Tensor) -> torch.Tensor:
    """TP 的层输出同步。单卡直通（零开销）。
    TP×EP 组合时在 TP 子组上做（不是 world group——否则 EP 的 rank 会被卷进来）。"""
    if not _TP_ENABLED or not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    if _TP_GROUP is not None and _TP_GROUP != dist.GroupMember.WORLD:
        dist.all_reduce(x, group=_TP_GROUP)
    else:
        dist.all_reduce(x)
    return x


class ColumnParallelLinear:
    """w13 的行切（=输出的中间维度切）：每卡输出 [out/TP, in]。
    用 mixin 方式给现有 BaseLinear 加切分能力（避免改 layers/linear.py 的单卡路径）。"""

    @staticmethod
    def shard_weight(w: torch.Tensor, tp_rank: int, tp_size: int, dim: int = 0) -> torch.Tensor:
        """按 dim 切权重给本 rank。整除校验防静默错误。
        ⚠ 必须 clone(): narrow().contiguous() 对连续切片返回 view，
        旧 storage 被共享 → 切分后显存不降（v6 实测 61GB 纹丝不动的根因）。"""
        assert w.shape[dim] % tp_size == 0, \
            f"TP{tp_size} 切不动 dim{dim}={w.shape[dim]}（必须整除）"
        per = w.shape[dim] // tp_size
        return w.narrow(dim, tp_rank * per, per).clone()


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
                                           per_kv * attn.head_dim).clone(),
            requires_grad=False)
        attn.v_proj.weight = __import__("torch").nn.Parameter(
            attn.v_proj.weight.data.narrow(0, kv_idx * per_kv * attn.head_dim,
                                           per_kv * attn.head_dim).clone(),
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
            __import__("torch").cat([gate_r, up_r], dim=1),   # cat 本身就是拷贝
            requires_grad=False)
        moe.w2 = __import__("torch").nn.Parameter(
            moe.w2.data.narrow(2, tp_rank * per_inter, per_inter).clone(),
            requires_grad=False)
        moe.inter = per_inter
        # router 全卡冗余（不切）——每卡都要算完整的 topk 路由
    torch.cuda.empty_cache()          # 释放被替换的全量权重
    return model


def shard_model_for_ep(model, ep_rank: int, ep_size: int):
    """EP 切分：每卡只保留自己一半专家的权重（其余置空）。
    router/embedding/attention 全卡冗余——输入 x 已被 TP allreduce 同步，
    每卡算本地专家的部分和再 all_reduce 聚合（DP-style EP，v6.4 换 all_to_all）。
    权重显存: 专家部分 /ep_size（30B 模型专家占 ~80% → 显存约省 40%/卡）。"""
    h = model.config
    E = h.num_experts
    assert E % ep_size == 0, f"专家数 {E} 切不动 EP{ep_size}"
    per = E // ep_size
    for layer in model.model.layers:
        moe = layer.mlp
        moe.expert_parallel = True
        moe.ep_rank = ep_rank
        moe.ep_size = ep_size
        moe.local_E = per
        moe.local_expert_start = ep_rank * per
        # 权重切片: 只留本地专家（显存真正释放）
        import torch as _t
        old13, old2 = moe.experts.w13, moe.experts.w2
        # 先解除 weight_loader（挂在 Parameter 属性上的方法引用会延长生命周期）
        for p_ in (old13, old2):
            if hasattr(p_, "weight_loader"):
                del p_.weight_loader
        moe.experts.w13 = _t.nn.Parameter(
            old13.data[ep_rank * per:(ep_rank + 1) * per].clone(),
            requires_grad=False)
        moe.experts.w2 = _t.nn.Parameter(
            old2.data[ep_rank * per:(ep_rank + 1) * per].clone(),
            requires_grad=False)
        del old13, old2
    torch.cuda.empty_cache()
    return model
