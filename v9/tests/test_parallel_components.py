"""v9.3 并行组件架构单元测试（纯 CPU，无 GPU/NCCL——用假 allreduce 模拟聚合）。

验证三层：
1. ColumnParallel/RowParallel：构造形状 + weight_loader 切片正确性
   （两 rank 的切片拼回 == 全量；单卡退化 == 全量）
2. GroupedExperts：EP 区间 + TP inter 切片的 loader 正确性
   （各 rank 本地专家拼回 == 全量专家）
3. 模型级：Qwen3MoeAttention 在 TP2 模拟下 per-rank 头数/形状正确

数学验证思路（无需真分布式）：把 N 个 rank 的本地权重各自 dequant/搬运出来，
在 CPU 上"手工 allreduce"（求和/拼接）后与全量参考对比。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from layers import parallel as P
from layers.linear import ColumnParallelLinear, RowParallelLinear
from layers.parallel import shard_bounds, configure_parallel


def set_tp(tp_size, tp_rank):
    """单进程模拟某个 tp rank（不初始化 dist——group 全 None, allreduce 直通）。"""
    configure_parallel(tp_size=tp_size, tp_rank=tp_rank, tp_group=None)


def test_shard_bounds():
    # 正常: 32 头, tp=8 → 每 rank 4 头
    assert shard_bounds(32, 8, 0) == (0, 4)
    assert shard_bounds(32, 8, 7) == (28, 4)
    # 复制: 4 kv 头(head_dim=8 → 32 维), tp=8, min_out=8 → n=4 份, rank 0/1 共享 head0
    assert shard_bounds(32, 8, 0, min_out=8) == (0, 8)
    assert shard_bounds(32, 8, 1, min_out=8) == (0, 8)    # 共享 head0
    assert shard_bounds(32, 8, 2, min_out=8) == (8, 8)    # head1
    assert shard_bounds(32, 8, 7, min_out=8) == (24, 8)   # head3
    print("shard_bounds: PASS")


def test_column_parallel():
    torch.manual_seed(0)
    N, K = 64, 32
    W = torch.randn(N, K)
    # tp=4: 每 rank 16 行
    shards = []
    for r in range(4):
        set_tp(4, r)
        lin = ColumnParallelLinear(K, N)
        assert lin.weight.shape == (16, K), lin.weight.shape
        lin.weight_loader(lin.weight, W)                  # 吃全量
        shards.append(lin.weight.data)
    recon = torch.cat(shards, dim=0)                       # 手工拼回
    assert torch.equal(recon, W), "列切拼回不等"
    # 单卡退化
    set_tp(1, 0)
    lin = ColumnParallelLinear(K, N)
    assert lin.weight.shape == (N, K)
    lin.weight_loader(lin.weight, W)
    assert torch.equal(lin.weight.data, W)
    # forward: 本 rank 切片乘
    set_tp(4, 2)
    lin = ColumnParallelLinear(K, N)
    lin.weight_loader(lin.weight, W)
    x = torch.randn(3, K)
    out = lin(x)
    ref = x @ W[32:48].t()
    assert torch.allclose(out, ref, atol=1e-6)
    print("ColumnParallelLinear: PASS（切回拼/单卡退化/forward）")


def test_row_parallel():
    torch.manual_seed(0)
    N, K = 32, 64
    W = torch.randn(N, K)
    outs = []
    for r in range(4):
        set_tp(4, r)
        lin = RowParallelLinear(K, N)
        assert lin.weight.shape == (N, 16), lin.weight.shape
        lin.weight_loader(lin.weight, W)
        x = torch.randn(3, 16)                             # 本 rank 的列切片输入
        outs.append(lin(x))                                # group=None → 直通（部分和）
    total = torch.stack(outs).sum(0)                       # 手工 allreduce
    x_full = None
    # 构造全量输入使各 rank 拿到自己的列段
    x_full = torch.randn(3, 64)
    set_tp(4, 0)
    lin0 = RowParallelLinear(K, N)
    lin0.weight_loader(lin0.weight, W)
    outs = []
    for r in range(4):
        set_tp(4, r)
        lin = RowParallelLinear(K, N)
        lin.weight_loader(lin.weight, W)
        outs.append(lin(x_full[:, r * 16:(r + 1) * 16]))
    total = torch.stack(outs).sum(0)
    ref = x_full @ W.t()
    assert torch.allclose(total, ref, atol=1e-5), \
        f"行切聚合不等: {(total - ref).abs().max()}"
    print("RowParallelLinear: PASS（列切装载/部分和聚合）")


def test_grouped_experts():
    from layers.moe import GroupedExperts
    torch.manual_seed(0)
    E, H, I = 8, 16, 32
    full_w13 = torch.randn(E, 2 * I, H)
    full_w2 = torch.randn(E, H, I)
    # EP=2: rank0 持专家 0-3, rank1 持 4-7
    experts_per_rank = {}
    for r in range(2):
        configure_parallel(tp_size=1, tp_rank=0, ep_size=2, ep_rank=r)
        ge = GroupedExperts(E, H, I)
        assert ge.num_experts == 4 and ge.e_start == r * 4
        assert ge.w13.shape == (4, 2 * I, H)
        # 装载全量 8 个专家（非本地跳过）
        for e in range(E):
            ge.weight_loader(ge.w13, full_w13[e], (e, 0))     # gate
            ge.weight_loader(ge.w13, full_w13[e][I:], (e, 1))  # up
            ge.weight_loader(ge.w2, full_w2[e], (e, None))
        experts_per_rank[r] = (ge.w13.data.clone(), ge.w2.data.clone())
    # 拼回验证
    w13_rec = torch.cat([experts_per_rank[0][0], experts_per_rank[1][0]], dim=0)
    w2_rec = torch.cat([experts_per_rank[0][1], experts_per_rank[1][1]], dim=0)
    assert torch.equal(w13_rec, full_w13), "EP w13 拼回不等"
    assert torch.equal(w2_rec, full_w2), "EP w2 拼回不等"

    # EP=2 × TP=2: inter 切半（w13 行 / w2 列）
    quarter = {}
    for er in range(2):
        for tr in range(2):
            configure_parallel(tp_size=2, tp_rank=tr, ep_size=2, ep_rank=er)
            ge = GroupedExperts(E, H, I)
            assert ge.num_experts == 4 and ge.inter == 16
            assert ge.w13.shape == (4, 32, H)
            for e in range(E):
                ge.weight_loader(ge.w13, full_w13[e], (e, 0))
                ge.weight_loader(ge.w13, full_w13[e][I:], (e, 1))
                ge.weight_loader(ge.w2, full_w2[e], (e, None))
            quarter[(er, tr)] = (ge.w13.data.clone(), ge.w2.data.clone())
    # tp 部分和拼回: 每 (er) 的两个 tr 的 w13 拼行 == full 对应专家
    for er in range(2):
        w13_t0, w2_t0 = quarter[(er, 0)]
        w13_t1, w2_t1 = quarter[(er, 1)]
        # w13: gate 半区 t0 行[0:16) + t1 行[16:32)
        gate = torch.cat([w13_t0[:, :16, :], w13_t1[:, :16, :]], dim=1)
        up = torch.cat([w13_t0[:, 16:, :], w13_t1[:, 16:, :]], dim=1)
        full_e = full_w13[er * 4:(er + 1) * 4]
        assert torch.equal(torch.cat([gate, up], dim=1), full_e), "TP×EP w13 拼回不等"
        w2 = torch.cat([w2_t0, w2_t1], dim=2)
        assert torch.equal(w2, full_w2[er * 4:(er + 1) * 4]), "TP×EP w2 拼回不等"
    print("GroupedExperts: PASS（EP 区间 / TP×EP 组合切回拼）")


def test_attention_shapes():
    """模型级：TP2 下 per-rank 头数（不跑 forward，只查构造）。"""
    from types import SimpleNamespace
    from models.qwen3_moe import Qwen3MoeAttention
    cfg = SimpleNamespace(head_dim=8, num_attention_heads=32, num_key_value_heads=4,
                          hidden_size=64, rms_norm_eps=1e-6)
    # 单卡
    set_tp(1, 0)
    attn = Qwen3MoeAttention(cfg, 0)
    assert attn.num_head == 32 and attn.num_kv_head == 4
    # TP2: q 16 头, kv 2 头
    set_tp(2, 0)
    attn = Qwen3MoeAttention(cfg, 0)
    assert attn.num_head == 16 and attn.num_kv_head == 2, (attn.num_head, attn.num_kv_head)
    assert attn.q_proj.weight.shape == (16 * 8, 64)
    assert attn.o_proj.weight.shape == (64, 16 * 8)
    # TP4（kv=4 可整切）
    set_tp(4, 1)
    attn = Qwen3MoeAttention(cfg, 0)
    assert attn.num_head == 8 and attn.num_kv_head == 1
    # TP8 + kv=4: 复制（rank0/1 共享 kv0）
    set_tp(8, 1)
    attn = Qwen3MoeAttention(cfg, 0)
    assert attn.num_head == 4
    assert attn.num_kv_head == 1, "kv 复制: 每 rank 至少 1 头"
    print("Qwen3MoeAttention: PASS（TP1/2/4/8 头数与形状，含 kv 复制）")




def test_gate_up_fc1():
    """fc1 的 gate|up 融合布局 × TP：直接均分 2I 会把 gate|up 劈开，
    gate_up=True 应各取 I/tp 再拼（chunk(2) 语义 rank 内成立）。"""
    from layers.activation import SiluAndMul
    torch.manual_seed(0)
    H, I = 16, 32
    W = torch.randn(2 * I, H)                     # [gate(32) | up(32)]
    x = torch.randn(3, H)
    outs = []
    for r in range(2):
        set_tp(2, r)
        fc1 = ColumnParallelLinear(H, 2 * I, gate_up=True)
        assert fc1.weight.shape == (I, H)         # [gate(I/2) | up(I/2)]
        fc1.weight_loader(fc1.weight, W)
        outs.append(SiluAndMul()(fc1(x)))         # [3, I/tp]
    # 聚合 = inter 维拼接（每 rank 已完成自己的 gate×up）
    full = torch.cat(outs, dim=-1)
    h = x @ W.t()
    g, u = h.chunk(2, -1)
    ref = torch.nn.functional.silu(g) * u
    assert torch.allclose(full, ref, atol=1e-5), \
        f"gate_up 切分数学错: {(full - ref).abs().max()}"
    print("ColumnParallel gate_up (fc1): PASS")


def test_vocab_parallel():
    """Embedding 行切（段外贡献 0 + 手工 allreduce）/ LmHead 行切（手工 gather 拼）。"""
    from layers.embedding import Embedding, LmHead
    torch.manual_seed(0)
    V, H = 64, 16
    W = torch.randn(V, H)
    ids = torch.tensor([0, 7, 33, 63, 40])
    outs = []
    for r in range(4):
        set_tp(4, r)
        emb = Embedding(V, H)
        assert emb.weight.shape == (16, H)
        emb.weight_loader(emb.weight, W)
        outs.append(emb(ids))                     # group=None → 直通（段外为 0）
    total = torch.stack(outs).sum(0)              # 手工 allreduce
    ref = W[ids]
    assert torch.allclose(total, ref, atol=1e-6), f"vocab embed 错: {(total-ref).abs().max()}"
    # LmHead: gather 拼接（x 固定——各 rank 输入相同）
    x = torch.randn(2, H)
    heads = []
    for r in range(4):
        set_tp(4, r)
        lm = LmHead(H, V)
        lm.weight_loader(lm.weight, W)
        heads.append(lm(x))
    full = torch.cat(heads, dim=-1)               # 手工 all_gather
    ref = x @ W.t()
    assert torch.allclose(full, ref, atol=1e-5), "vocab lm_head 拼接错"
    # 单卡退化
    set_tp(1, 0)
    emb = Embedding(V, H); emb.weight_loader(emb.weight, W)
    assert torch.allclose(emb(ids), W[ids])
    print("VocabParallel Embedding/LmHead: PASS（切分聚合/单卡退化）")


def test_dense_qwen3_shapes():
    """dense 模型（qwen3.py）的 TP 构造。"""
    from types import SimpleNamespace
    from models.qwen3 import QwenAttention, QwenMLP
    cfg = SimpleNamespace(head_dim=8, num_attention_heads=16, num_key_value_heads=4,
                          hidden_size=32, intermediate_size=64,
                          rms_norm_eps=1e-6, hidden_act="silu")
    set_tp(2, 1)
    attn = QwenAttention(cfg, 0)
    assert attn.num_head == 8 and attn.num_kv_head == 2
    mlp = QwenMLP(cfg)
    assert mlp.fc1.weight.shape == (64, 32), mlp.fc1.weight.shape   # gate32|up32 → 各取16
    assert mlp.fc2.weight.shape == (32, 32)
    set_tp(1, 0)
    mlp1 = QwenMLP(cfg)
    assert mlp1.fc1.weight.shape == (128, 32)
    print("dense Qwen3 (qwen3.py): PASS（TP2/单卡 MLP+attention 形状）")


if __name__ == "__main__":
    test_shard_bounds()
    test_column_parallel()
    test_row_parallel()
    test_grouped_experts()
    test_attention_shapes()
    test_gate_up_fc1()
    test_vocab_parallel()
    test_dense_qwen3_shapes()
    print("\n===== v9.3b 并行组件架构单测: ALL PASS =====")
