# v6/engine/_legacy_tp.py
"""v6-v8 旧引擎快照的最小 TP 工具（从已删除的 engine/parallel.py 原样摘出）。

⚠ 仅服务于历史快照的两个仍可运行的验证路径：
  1. v6-v8 引擎主路径的模型 forward（qwen3_moe.py 显式调 tp_allreduce——
     v6 时代的"通信在模型层"设计，v9.3 起被组件架构取代）；
  2. v8/tests/run_dcp_tp.py 的 TP phase（DCP 三相位验证，2026-11-01 实测
     base 16.5 / TP2 7.7 / TP2×DCP2 3.6 tok/s 的入口）。

v9 的活代码【不 import 本文件】——并行组件在 layers/parallel.py +
layers/linear.py（ColumnParallel/RowParallel，构造即分片）。
这是"全量加载后就地切分"的 deprecated 模式（narrow view 陷阱的来源，
见 ENGINEERING_NOTES 2.0），保留只为历史快照可复现，新代码勿用。
"""
import torch
import torch.distributed as dist

# 旧全局开关（v6 语义）
_TP_ENABLED = False
_TP_GROUP = None


def set_tp_enabled(v: bool):
    global _TP_ENABLED
    _TP_ENABLED = v


def set_tp_group(g):
    global _TP_GROUP
    _TP_GROUP = g


def tp_allreduce(x: torch.Tensor) -> torch.Tensor:
    """层输出 allreduce（单卡/未启用/未初始化直通）。"""
    if not _TP_ENABLED or not dist.is_available() or not dist.is_initialized():
        return x
    if _TP_GROUP is not None and _TP_GROUP != dist.GroupMember.WORLD:
        dist.all_reduce(x, group=_TP_GROUP)
    else:
        dist.all_reduce(x)
    return x


def _shard_weight(w: torch.Tensor, tp_rank: int, tp_size: int, dim: int) -> torch.Tensor:
    """沿 dim 均分取本 rank 片。⚠ 必须 clone()：narrow().contiguous() 对
    连续切片返回 view（共享 storage）——61GB 显存幽灵的根因。"""
    per = w.shape[dim] // tp_size
    return w.narrow(dim, tp_rank * per, per).clone()


def shard_model_for_tp(model, tp_rank: int, tp_size: int):
    """把已加载完整权重的模型就地切分到本 rank（deprecated：内存换简单）。

    kv 头数 < TP 时（如 kv=4 < TP8）：kv 复制不切（每 rank 持有一份复制），
    q/o 仍切——多余 rank 重复算同一些 kv 头，正确性不受影响。
    切分点：attention q/k/v/o（按头）、MoE w13/w2（gate|up 各半，切缝不劈）、
    router 全卡冗余。
    """
    h = model.config
    assert h.num_attention_heads % tp_size == 0
    per_q = h.num_attention_heads // tp_size
    per_kv = max(1, h.num_key_value_heads // tp_size)
    per_inter = h.moe_intermediate_size // tp_size
    I_total = h.moe_intermediate_size

    for layer in model.model.layers:
        attn = layer.self_attn
        # q_proj [q_size, H] → 本 rank 的头 [per_q*hd, H]
        attn.q_proj.weight = torch.nn.Parameter(
            _shard_weight(attn.q_proj.weight.data, tp_rank, tp_size, 0),
            requires_grad=False)
        # kv：tp_rank 映射到 kv 切片（TP>kv 时多个 rank 映射同一 kv 切片 = 复制）
        n_kv_shards = max(1, h.num_key_value_heads // per_kv)
        kv_idx = tp_rank * n_kv_shards // tp_size
        attn.k_proj.weight = torch.nn.Parameter(
            attn.k_proj.weight.data.narrow(0, kv_idx * per_kv * attn.head_dim,
                                           per_kv * attn.head_dim).clone(),
            requires_grad=False)
        attn.v_proj.weight = torch.nn.Parameter(
            attn.v_proj.weight.data.narrow(0, kv_idx * per_kv * attn.head_dim,
                                           per_kv * attn.head_dim).clone(),
            requires_grad=False)
        # o_proj [H, q_size] 列切（输入是本 rank 的头）
        attn.o_proj.weight = torch.nn.Parameter(
            _shard_weight(attn.o_proj.weight.data, tp_rank, tp_size, 1),
            requires_grad=False)
        # 头数视角同步——外层 Attention 和内层 triton Attention 都要改
        # （kernel 的 GQA 映射读内层 num_kv_heads）
        attn.num_head = per_q
        attn.num_kv_head = per_kv
        attn.q_size = per_q * attn.head_dim
        attn.kv_size = per_kv * attn.head_dim
        if hasattr(attn, "attn"):
            attn.attn.num_heads = per_q
            attn.attn.num_kv_heads = per_kv
        # MoE: w13 布局 [E, gate(I)|up(I), H]——每卡拿 gate 一半 + up 一半
        # （forward 的 chunk(2) 切 gate|up 后各自是 I/TP），不是整段行切！
        moe = layer.mlp.experts
        w13 = moe.w13.data
        gate_r = w13[:, tp_rank * per_inter:(tp_rank + 1) * per_inter, :]
        up_r = w13[:, I_total + tp_rank * per_inter:
                   I_total + (tp_rank + 1) * per_inter, :]
        moe.w13 = torch.nn.Parameter(torch.cat([gate_r, up_r], dim=1),
                                     requires_grad=False)   # cat 本身即拷贝
        moe.w2 = torch.nn.Parameter(
            moe.w2.data.narrow(2, tp_rank * per_inter, per_inter).clone(),
            requires_grad=False)
        moe.inter = per_inter
    torch.cuda.empty_cache()          # 释放被替换的全量权重
    return model
