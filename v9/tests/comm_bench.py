# v6/comm_bench.py
"""实验一：通信量微基准——TP allreduce vs EP all-to-all 随 batch 的曲线。

验证 PARALLEL_ANALYSIS §2 的公式：
  TP 每层通信 ∝ b×s×h（全激活 allreduce）
  EP 每层通信 ∝ tokens×h（token 分发）
用 Qwen3-Omni thinker 的真实尺寸（hidden=2048, 48 层）模拟每步的总通信成本。

用法: python comm_bench.py            # 双卡本机
      python comm_bench.py --seq 512  # prefill 场景的 s
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

HIDDEN = 2048
LAYERS = 48


def bench_op(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000   # µs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--seq", type=int, default=1, help="decode: 1; prefill 用 prompt 长度")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 2))
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    dev = f"cuda:{rank}"

    print(f"{'batch':>6} {'TP/步(ms)':>10} {'EP/步(ms)':>10} {'TP 每层字节':>14} {'EP 每层字节':>14}")
    for b in [int(x) for x in args.batches.split(",")]:
        tokens = b * args.seq
        # TP: 每层 2 次 allreduce [b, s, h] bf16
        tp_tensor = torch.randn(b, args.seq, HIDDEN, dtype=torch.bfloat16, device=dev)
        t_us = bench_op(lambda: [dist.all_reduce(tp_tensor) for _ in range(2)])
        tp_ms = t_us * LAYERS / 1000
        tp_bytes = 2 * 2 * tokens * HIDDEN * 2   # 2次 × ring系数2 × 数据量

        # EP: 每层 2 次 all-to-all [b*s, h]（dispatch+combine，各卡发 1/world）
        ep_tensor = torch.randn(tokens * HIDDEN, dtype=torch.bfloat16, device=dev)
        chunk = tokens * HIDDEN // world

        def ep_step():
            for _ in range(2):
                out = torch.empty_like(ep_tensor)
                dist.all_to_all_single(out, ep_tensor)
        e_us = bench_op(ep_step)
        ep_ms = e_us * LAYERS / 1000
        ep_bytes = 2 * tokens * HIDDEN * 2 * 2    # dispatch+combine 各传全量(均摊)

        if rank == 0:
            print(f"{b:>6} {tp_ms:>10.3f} {ep_ms:>10.3f} {tp_bytes/1e6:>12.1f}MB {ep_bytes/1e6:>12.1f}MB", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
