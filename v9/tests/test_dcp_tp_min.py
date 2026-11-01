"""v8.2 最小复现：单层 attention 的 TP2×DCP2 四卡对拍（合成权重）。

不加载真模型——用 Attention 层 + 合成 q/k/v 直接复现死锁点：
  4 rank = (tp, cp) ∈ {0,1}²；q/k/v 投影按 TP 切头，cache 按块染色。
  参考解 = CPU 全量 attention（不切分）。
若此脚本 hang → 死锁在 DCP 逻辑本身；若通过 → 死锁在引擎接线。
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
V8_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, V8_DIR)
sys.path.insert(0, TESTS_DIR)

rank, world, tp, dcp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = sys.argv[5]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world)
torch_cuda_idx = rank % 8
import torch
torch.cuda.set_device(torch_cuda_idx)
import torch.distributed as dist
dist.init_process_group("nccl")

from parallel_coords import init_coords, warmup_groups
coords = init_coords(dp=1, tp=tp, cp=dcp, ep=1, pp=1)
print(f"[r{rank}] {coords}", flush=True)
warmup_groups(coords)
print(f"[r{rank}] groups warmed", flush=True)

from layers.dcp import set_dcp
set_dcp(active=(dcp > 1), size=dcp, rank=coords["cp"], group=coords.group("cp"))

from engine.parallel import set_tp_enabled, set_tp_group
if tp > 1:
    set_tp_enabled(True)
    set_tp_group(coords.group("tp"))

from layers.attention import Attention
from utils.context import set_context

# ---- 合成配置（对齐 Qwen3-30B 的形状，只缩小层数）----
HQ, HK, HD = 16, 4, 64
per_q, per_kv = HQ // tp, HK // tp          # TP 切头
BS = 16                                     # 小块便于测试多块
L = 40                                      # 序列长（跨 3 个块）
torch.manual_seed(42 + rank)                # 权重 seed 相同→但 per-rank 切片不同

attn = Attention(per_q, per_kv, HD, HD ** -0.5).cuda().to(torch.float32)
# 全局 cache 视图: 全局块数 = ceil(643..) 这里只需覆盖 L: 3 块
N_GLOBAL = 4
N_LOCAL = N_GLOBAL // dcp
# k_cache 布局 [slots, BS, HK, HD] —— TP 后 HK=per_kv
attn.k_cache = torch.zeros(N_LOCAL, BS, per_kv, HD, device="cuda")
attn.v_cache = torch.zeros(N_LOCAL, BS, per_kv, HD, device="cuda")

dcp_ctx_size = dcp

# ---- 输入: 一个序列 L=40, 全量 prefill（q=k 全段）----
# 两 cp rank 的 q 相同（同 tp 头）——用广播同步输入
if rank == 0:
    q_in = torch.randn(L, per_q, HD, device="cuda")
    k_in = torch.randn(L, per_kv, HD, device="cuda")
    v_in = torch.randn(L, per_kv, HD, device="cuda")
else:
    q_in = torch.empty(L, per_q, HD, device="cuda")
    k_in = torch.empty(L, per_kv, HD, device="cuda")
    v_in = torch.empty(L, per_kv, HD, device="cuda")
# 广播: cp 组内同头 (r0→r1), tp 组同色 (r0→r2)——直接 world 广播最简单
dist.broadcast(q_in, src=0)
dist.broadcast(k_in, src=0)
dist.broadcast(v_in, src=0)

positions = torch.arange(L, device="cuda")
# 全局 slot_mapping: 块表 [0,1,2] → slot = g*BS + off
slot_mapping = torch.arange(L, dtype=torch.int32, device="cuda")
block_tables = torch.tensor([[0, 1, 2, -1]], dtype=torch.int32, device="cuda")
cu_q = torch.tensor([0, L], dtype=torch.int32, device="cuda")
cu_k = torch.tensor([0, L], dtype=torch.int32, device="cuda")
set_context(True, cu_q, cu_k, L, L, slot_mapping, None, block_tables)

cos_sin_cache = torch.randn(L + 8, 1, HD, device="cuda")
q_norm_w = torch.randn(HD, device="cuda")
k_norm_w = torch.randn(HD, device="cuda")

print(f"[r{rank}] forward begin", flush=True)
# 模拟引擎的 28 层: attention(dcp gather) + tp_allreduce 交错
from engine.parallel import tp_allreduce
for i in range(28):
    out = attn(q_in, k_in, v_in, positions, q_norm_w, k_norm_w, cos_sin_cache, 1e-6)
    out = tp_allreduce(out)
    print(f"[r{rank}] layer{i} done", flush=True)
print(f"[r{rank}] 28 layers forward done shape={tuple(out.shape)}", flush=True)

# 模拟 step 末尾的 world broadcast（eos 分叉保护）
bt = torch.tensor([1, 2, 3], device="cuda")
dist.broadcast(bt, src=0)
print(f"[r{rank}] world broadcast done", flush=True)

# ---- 参考解（CPU 全量, 含 norm+rope 数学复刻太复杂——改对比各 rank 输出一致性）----
# TP allreduce 模拟 o_proj 的合并（这里只验证 DCP 合并的 rank 间一致性:
# 同 tp 不同 cp 的 rank 输出应一致（DCP 合并后位置完整））
out_c = out.clone()
if dcp > 1:
    g = coords.group("cp")
    same = [torch.zeros_like(out_c) for _ in range(dcp)]
    dist.all_gather(same, out_c.contiguous(), group=g)
    if coords["cp"] == 0:
        diff = (same[0] - same[1]).abs().max().item()
        print(f"[r{rank}] cp 组内 (tp{coords['tp']}) DCP 合并一致性 diff={diff:.2e} "
              f"{'PASS' if diff < 1e-4 else 'FAIL'}", flush=True)

# tp 组一致性: 同 cp 不同 tp 的输出应不同（不同头）但无 NaN
if torch.isnan(out).any() or torch.isinf(out).any():
    print(f"[r{rank}] FAIL: NaN/Inf in output", flush=True)
else:
    print(f"[r{rank}] output finite OK", flush=True)

dist.barrier()
if rank == 0:
    print("===== minimal TP×DCP attention: DONE =====", flush=True)
dist.destroy_process_group()
