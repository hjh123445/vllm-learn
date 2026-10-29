"""v9 设计先行实验：RTN 量化误差实测（真实 Qwen3-30B 权重）+ 落地路径 B1 的性能陷阱。

回答三个设计问题（数据说话）：
1. 粒度选择：per-group g=32/64/128/256 vs per-channel vs per-tensor，int4/int8 的误差
2. 存储收益：int4+g128 的实际压缩比（含 scale 开销）
3. 路径 B1（存 int4、forward 时 unpack→bf16 matmul）的速度真相——显存收益 vs 带宽真相
"""
import json
import time

import torch
from safetensors import safe_open

PATH = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
idx = json.load(open(f"{PATH}/model.safetensors.index.json"))
wm = idx["weight_map"]


def load_tensor(name):
    f = safe_open(f"{PATH}/{wm[name]}", framework="pt", device="cpu")
    return f.get_tensor(name)


def rtn_quant(w: torch.Tensor, bits: int, group: int):
    """对称 RTN：沿最后一维（输入维）分组。group=0 → per-channel。
    返回 (rel_mse, max_rel_err)。int4 用 [-8,7]，int8 用 [-127,127]。"""
    qmax = 2 ** (bits - 1) - 1
    K = w.shape[-1]
    g = group if group else K
    pad = (-K) % g
    wp = torch.nn.functional.pad(w.float(), (0, pad)) if pad else w.float()
    wv = wp.reshape(*wp.shape[:-1], -1, g)
    s = wv.abs().amax(-1).clamp(min=1e-10) / qmax          # [.., n_groups]
    q = (wv / s.unsqueeze(-1)).round().clamp_(-qmax - 1, qmax)
    deq = (q * s.unsqueeze(-1)).reshape(*wp.shape)
    if pad:
        deq = deq[..., :K]
    err = deq - w.float()
    rel_mse = (err ** 2).mean().item() / (w.float() ** 2).mean().item()
    max_rel = (err.abs().max() / w.float().abs().max()).item()
    return rel_mse, max_rel


# ---- 1. 选真实张量：attention 投影 + 一个专家的三个矩阵 ----
names = sorted(wm.keys())
targets = {}
for n in names:
    if "layers.10.self_attn.q_proj.weight" in n:
        targets["q_proj [N,H]"] = n
    if "layers.10.mlp.experts.0.gate_proj.weight" in n:
        targets["expert gate [I,H]"] = n
    if "layers.10.mlp.experts.0.down_proj.weight" in n:
        targets["expert down [H,I]"] = n
    if len(targets) == 3:
        break

print("=" * 72)
print("1. RTN 量化误差（真实 Qwen3-30B-A3B 权重，对称量化，沿输入维分组）")
print("=" * 72)
header = f"{'tensor':<20}{'int4 g=32':>12}{'int4 g=64':>12}{'int4 g=128':>12}{'int4 g=256':>12}{'int4 ch':>10}{'int8 ch':>10}{'int4 tensor':>12}"
print(header)
for label, name in targets.items():
    w = load_tensor(name)
    row = f"{label:<20}"
    for bits, g in [(4, 32), (4, 64), (4, 128), (4, 256), (4, 0), (8, 0)]:
        rel, _ = rtn_quant(w, bits, g)
        row += f"{rel:>11.2e} "
    print(row)

print()
print("=" * 72)
print("2. 存储收益（16bit 基准 → 量化的有效位宽）")
print("=" * 72)
for g, note in [(32, ""), (64, ""), (128, "AWQ/GPTQ 标准"), (256, "")]:
    bits_eff = 4 + 16 / g            # int4 + fp16 scale / g 个权重
    print(f"int4 g={g:<4} {note:<16} 有效位宽 {bits_eff:.3f} bit → 压缩 {16/bits_eff:.2f}x")
print(f"int8 per-channel (K=2048)  有效位宽 {8+16/2048:.3f} bit → 压缩 {16/(8+16/2048):.2f}x")

print()
print("=" * 72)
print("3. 路径 B1 性能真相：int4 存储 + forward unpack→bf16 matmul（GPU 实测）")
print("=" * 72)
N, K = 2048, 2048     # 典型投影形状
dev = "cuda"
torch.manual_seed(0)
w_bf = (torch.randn(N, K) * 0.02).to(torch.bfloat16).to(dev)

# int4 打包: [N, K/2] uint8（低半字节=偶数位，高半字节=奇数位）+ scale [N, K/g]
g = 128
qmax = 7
s = w_bf.float().reshape(N, -1, g).abs().amax(-1).clamp(min=1e-10) / qmax   # [N, K/g]
q = (w_bf.float().reshape(N, -1, g) / s.unsqueeze(-1)).round().clamp_(-8, 7).to(torch.int8)
wq_packed = (q[..., 0::2].to(torch.uint8) | (q[..., 1::2].to(torch.uint8) << 4)).reshape(N, K // 2).to(dev)
s_dev = s.to(dev).to(torch.bfloat16)


def b1_forward(x, wq_packed, s, g):
    """B1: unpack → dequant → bf16 matmul（无融合 kernel 的朴素路径）。"""
    N, K2 = wq_packed.shape
    K = K2 * 2
    lo = (wq_packed & 0xF).to(torch.bfloat16)
    hi = (wq_packed >> 4).to(torch.bfloat16)
    q = torch.stack([lo, hi], dim=-1).reshape(N, K)          # interleaved 还原
    deq = q.reshape(N, -1, g) * s.unsqueeze(-1)              # 反量化
    return x @ deq.reshape(N, K).t()


for M, tag in [(8, "decode bs=8"), (4096, "prefill 4096")]:
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)

    def bench(fn, n=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000

    t_bf = bench(lambda: x @ w_bf.t())
    t_b1 = bench(lambda: b1_forward(x, wq_packed, s_dev, g))
    # 数值一致性（B1 与 bf16 直接乘的差异 = 纯量化误差, 不应有实现误差）
    ref = (x.float() @ w_bf.float().t())
    out = b1_forward(x, wq_packed, s_dev, g).float()
    diff = (out - ref).abs().max().item()
    print(f"M={M:<5}({tag:<13}) bf16 matmul {t_bf:7.3f}ms | B1 unpack+matmul {t_b1:7.3f}ms "
          f"(x{t_b1/t_bf:.2f}) | 数值 diff {diff:.2e}")

print()
print("结论预期: B1 显存省 ~3.9x 但【更慢】——多出了 unpack 的读写流量")
print("(int4 读 + bf16 写 + bf16 读 vs 一次 bf16 读)。速度收益必须靠 B2 融合 kernel。")
