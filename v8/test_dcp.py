"""v8.1 DCP2 端到端验证：双卡 DCP decode vs 单卡全量 decode 逐 token 一致。

架构（建立在 v7 引擎上）：
  - 两个进程各持一半 KV cache（块染色）
  - decode 步: 每卡对自己的块算 partial+LSE → all_gather → 本地 LSE 合并
  - 采样用合并后的 logits（两卡一致）
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

rank = int(sys.argv[1])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29910"
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = "2"
torch.cuda.set_device(rank)
import torch.distributed as dist
dist.init_process_group("nccl")

from parallel_coords import init_coords
coords = init_coords(dp=1, tp=1, cp=2, ep=1, pp=1)     # DCP2
print(f"[r{rank}] {coords}", flush=True)

from layers.dcp import DCPCacheCoordinator, partial_attention_with_lse, lse_merge_stack

# ============ 模拟 DCP 的 decode attention（单层, 合成权重）============
torch.manual_seed(42)   # 两卡同权重
H, HQ, HK, D = 8, 16, 4, 64
scale = D ** -0.5
Wq = torch.randn(HQ * D, H, device="cuda") * 0.1
Wk = torch.randn(HK * D, H, device="cuda") * 0.1
Wv = torch.randn(HK * D, H, device="cuda") * 0.1

# 序列的 KV cache（逻辑上完整 L=64 位置, 物理上块染色分片）
L = 64
x_hist = torch.randn(L, H, device="cuda") * 0.5
K_full = (x_hist @ Wk.t()).view(L, HK, D)
V_full = (x_hist @ Wv.t()).view(L, HK, D)

# 块大小 8 → 8 个块; 染色: block % 2
coord = DCPCacheCoordinator(2, rank)
blocks = list(range(L // 8))
my_blocks = [b for b in blocks if coord.owns(b)]
# 本 rank 持有的位置（按块展开）
my_pos = []
for b in my_blocks:
    my_pos += list(range(b * 8, b * 8 + 8))
K_local = K_full[my_pos]      # [L/2, HK, D]
V_local = V_full[my_pos]

# ---- 单卡参考（每卡独立算全量, 验证一致性用）----
def full_decode(q_hidden):
    q = (q_hidden @ Wq.t()).view(-1, HQ, D)
    k2 = K_full.repeat_interleave(HQ // HK, dim=1)  # [L, HQ, D]
    v2 = V_full.repeat_interleave(HQ // HK, dim=1)
    s = torch.einsum("qhd,lhd->hql", q, k2) * scale
    p = s.softmax(-1)
    return torch.einsum("hql,lhd->qhd", p, v2)

# ---- DCP decode（每卡 partial + LSE 合并）----
def dcp_decode(q_hidden):
    q = (q_hidden @ Wq.t()).view(-1, HQ, D)
    k2 = K_local.repeat_interleave(HQ // HK, dim=1)  # [L/2, HQ, D]
    v2 = V_local.repeat_interleave(HQ // HK, dim=1)
    s = torch.einsum("qhd,lhd->hql", q, k2) * scale  # [HQ, 1, L/2]
    m = s.max(-1).values                              # [HQ, 1]
    p = torch.exp(s - m[..., None])
    l = p.sum(-1)
    part = torch.einsum("hql,lhd->qhd", p / l[..., None], v2)  # [1, HQ, D]
    # 调整形状到 [b=1, h, d] 并 gather
    part = part.permute(1, 0, 2)                      # [1, HQ, D] → [HQ,1,D]? 统一为 [b,h,d]
    part = part.reshape(1, HQ, D)
    m = m.reshape(1, HQ)
    l = l.reshape(1, HQ)
    # all_gather: [2, b, h, d] / [2, b, h]
    parts = [torch.zeros_like(part) for _ in range(2)]
    dist.all_gather(parts, part.contiguous())
    ms = [torch.zeros_like(m) for _ in range(2)]
    ls = [torch.zeros_like(l) for _ in range(2)]
    dist.all_gather(ms, m.contiguous())
    dist.all_gather(ls, l.contiguous())
    return lse_merge_stack(torch.stack(parts), torch.stack(ms), torch.stack(ls))

# ---- 逐 token 一致性验证（模拟 8 步 decode）----
torch.manual_seed(rank + 100)
ok = True
for step in range(8):
    q_hidden = torch.randn(1, H, device="cuda") * 0.5   # 两卡同输入?
    # 不同 seed 会不同——用固定同步的输入
    if rank == 0:
        q_hidden = torch.randn(1, H, device="cuda") * 0.5
        dist.broadcast(q_hidden, src=0)
    else:
        dist.broadcast(q_hidden, src=0)

    ref = full_decode(q_hidden)                        # 每卡独立的全量参考
    out = dcp_decode(q_hidden)                         # DCP 合并结果
    diff = (out - ref).abs().max().item()
    if rank == 0:
        print(f"step {step}: DCP vs 全量 max diff = {diff:.2e} "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}", flush=True)
        ok &= diff < 1e-4

if rank == 0:
    print("\n===== v8.1 DCP2 端到端:", "ALL PASS" if ok else "FAILED", "=====")

# 显存收益报告（逻辑统计——实际块染色在引擎里做）
if rank == 0:
    full_kb = K_full.numel() * 4 / 1024
    local_kb = K_local.numel() * 4 / 1024
    print(f"KV cache 显存: 全量 {full_kb:.0f}KB → 本 rank {local_kb:.0f}KB "
          f"(持有 {len(my_blocks)}/{len(blocks)} 块 = {len(my_pos)}/{L} 位置)")

dist.destroy_process_group()
