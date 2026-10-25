# benchmark/capacity.py
"""容量测试：显存占用 / 池子大小 / 最大并发请求数。

问题：这个引擎吃多少显存？池子多大、怎么来的？同样的显存能并发多少请求？
对比三个引擎：v0/v1（拍脑袋预分配）vs v2（warmup 量出来再分配）。

用法: python benchmark/capacity.py --num-requests 24 --batch-size 16 --gpu-mem-util 0.3
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def worker(args):
    sys.path.insert(0, os.path.join(ROOT, args.engine))
    import time
    import torch
    from transformers import AutoTokenizer
    import harness
    from harness import MODEL_PATH, build_workload_impl, run_v0, run_v1, run_v2
    from samplingParams import SamplingParams
    from engine.scheduler import Scheduler
    from engine.model_runner import ModelRunner

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos = {tok.eos_token_id}
    ekw = {"gpu_memory_utilization": args.gpu_mem_util} if args.engine == "v2" else {}

    # ---- 量权重和池子（单独起一个引擎量完就删，别算进后面的峰值）----
    r = ModelRunner(MODEL_PATH, batch_size=args.batch_size,
                    max_seq_len=4096, eos=tok.eos_token_id, **ekw)
    if args.engine == "v2":
        pool_bytes = r.kv.pool_bytes()
        blocks, block_mb = r.kv.max_num_blocks, r.kv.block_bytes / 1e6
    else:
        pool_bytes = sum(m.cache.k.numel() * m.cache.k.element_size() * 2
                         for m in r.model.modules() if hasattr(m, "cache"))
        blocks = block_mb = None
    weights_gb = (torch.cuda.memory_allocated() - pool_bytes) / 1e9
    pool_gb = pool_bytes / 1e9
    del r, pool_bytes
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---- 负载跑一遍，量峰值显存与实际并发（统一 max_tokens，聚焦 KV 占用）----
    harness.MAX_TOKEN_TIERS = [args.max_tokens]
    prompt_ids, sps = build_workload_impl(args.num_requests, tok, SamplingParams)
    t0 = time.perf_counter()
    kw = dict(prompt_ids=prompt_ids, sps=sps, batch_size=args.batch_size, max_seq_len=4096)
    if args.engine == "v0":
        res = run_v0(ModelRunner, Scheduler, eos_ids=eos,
                     pad_id=tok.pad_token_id or tok.eos_token_id, **kw)
    elif args.engine == "v1":
        res = run_v1(ModelRunner, Scheduler, eos_ids=eos, **kw)
    else:
        runner = ModelRunner(MODEL_PATH, batch_size=args.batch_size,
                             max_seq_len=4096, eos=tok.eos_token_id, **ekw)
        res = run_v2(runner, Scheduler(eos_ids=eos, max_seq_len=4096,
                                       max_batch_size=args.batch_size, kv=runner.kv),
                     eos_ids=eos, **kw)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    out = {
        "engine": args.engine,
        "weights_gb": round(weights_gb, 2),
        "pool_gb": round(pool_gb, 2),
        "pool_blocks": blocks,
        "block_mb": round(block_mb, 2) if block_mb else None,
        "peak_gb": round(peak_gb, 2),
        "peak_concurrent": max((s["n_running"] for s in res["steps"]), default=0),
        "total_out_tokens": sum(r["n_tokens"] for r in res["requests"]),
        "wall_s": round(time.perf_counter() - t0, 2),
        "tok_per_s": round(sum(r["n_tokens"] for r in res["requests"]) / res["total_time"], 1),
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
    }
    if args.engine == "v2":
        per_req_blocks = (res["requests"][0]["n_tokens"] + 30) // 16 + 1
        out["kv_per_req_mb"] = round(per_req_blocks * (block_mb or 0), 2)
        out["theoretical_1k_concurrent"] = (blocks // 65 if blocks else 0)
    print(json.dumps(out, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")          # 不传 = 三个引擎都跑（驱动模式）
    ap.add_argument("--num-requests", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--gpu-mem-util", type=float, default=0.3)
    args = ap.parse_args()
    if args.engine:
        return worker(args)

    rows = []
    for e in ("v0", "v1", "v2"):
        r = subprocess.run([sys.executable, __file__, "--engine", e,
                            "--num-requests", str(args.num_requests),
                            "--batch-size", str(args.batch_size),
                            "--max-tokens", str(args.max_tokens),
                            "--gpu-mem-util", str(args.gpu_mem_util)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{e} 崩溃: {r.stderr.strip().splitlines()[-1]}")
            continue
        rows.append(json.loads(r.stdout.strip().splitlines()[-1]))

    out_path = os.path.join(HERE, "results", "capacity.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    hdr = ["engine", "weights_gb", "pool_gb", "pool_blocks", "peak_gb",
           "peak_concurrent", "tok_per_s", "batch_size"]
    print("  ".join(f"{h:>14}" for h in hdr))
    for r in rows:
        print("  ".join(f"{str(r.get(h, '-')):>14}" for h in hdr))
    extra = [{k: v for k, v in r.items() if k not in hdr} for r in rows]
    print(json.dumps(extra, ensure_ascii=False, indent=1))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
