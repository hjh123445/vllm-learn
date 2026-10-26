# benchmark/stress.py
"""压力测试：请求按固定间隔到达（模拟在线流量），测时延分位数与吞吐。

与 speed.py 的区别：speed 是"一批同时全到"（burst），stress 是"陆续到达"——
这才是考验准入/驱逐/抢占的场景。

用法: python benchmark/stress.py --engines v1 v2 --num-requests 32 --arrive-interval 0.25
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
    from harness import MODEL_PATH, QUESTIONS
    from samplingParams import SamplingParams
    from engine.model_runner import ModelRunner
    from engine.scheduler import Scheduler

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos = {tok.eos_token_id}
    ekw = {"gpu_memory_utilization": args.gpu_mem_util} if args.engine == "v2" else {}
    runner = ModelRunner(MODEL_PATH, batch_size=args.batch_size,
                         max_seq_len=4096, eos=tok.eos_token_id, **ekw)
    sch = (Scheduler(eos_ids=eos, max_seq_len=4096, max_batch_size=args.batch_size,
                      kv=runner.kv) if args.engine == "v2"
           else Scheduler(eos_ids=eos, max_seq_len=4096, max_batch_size=args.batch_size))

    # 到达计划：第 i 个请求在 i × interval 秒到
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    pending = []
    for i in range(args.num_requests):
        q = QUESTIONS[i % len(QUESTIONS)]
        text = tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
        pending.append((i * args.arrive_interval, tok.encode(text, add_special_tokens=False), sp))

    arrive_t, first_t, finish_t = {}, {}, {}
    t0 = time.perf_counter()
    while pending or sch.busy:
        t = time.perf_counter() - t0
        while pending and pending[0][0] <= t:
            _, ids, s = pending.pop(0)
            req = sch.add(ids, s)
            arrive_t[req.id] = t

        if args.engine == "v1":
            # v1: 混合批次（新请求 prefill + 在途 decode 同一个 forward）
            new = sch.admit()
            new_ids = {r.id for r in new}
            running = [r for r in sch.running if r.id not in new_ids]
            if new or running:
                nt = runner.step_mixed(new, running, 0.0, 1.0)
                now = time.perf_counter()
                for req, token in zip(new + running, nt.tolist()):
                    was = req.finished
                    req.step(token)
                    if req in new:
                        first_t[req.id] = now - t0
                    elif not was and req.finished:
                        finish_t[req.id] = now - t0
                sch.evict()
        else:
            # v2: scheduler 全权管理（schedule 给什么跑什么，抢占在 schedule 内部）
            batch, is_prefill = sch.schedule()
            if batch:
                nt = runner.step(batch, is_prefill, 0.0, 1.0)
                now = time.perf_counter()
                for req, token in zip(batch, nt.tolist()):
                    was = req.finished
                    req.step(token)
                    if is_prefill:
                        first_t[req.id] = now - t0
                    elif not was and req.finished:
                        finish_t[req.id] = now - t0
                sch.evict()

        if not sch.running and pending:
            time.sleep(max(0.0, pending[0][0] - (time.perf_counter() - t0)))

    total = time.perf_counter() - t0
    ttft = sorted((first_t[r] - arrive_t[r]) * 1e3 for r in first_t)
    e2e = sorted((finish_t.setdefault(r, first_t[r]) - arrive_t[r]) * 1e3 for r in arrive_t)

    def pct(xs, p):
        return xs[min(int(len(xs) * p), len(xs) - 1)]

    out = {
        "engine": args.engine,
        "num_requests": args.num_requests,
        "arrive_interval_s": args.arrive_interval,
        "max_tokens": args.max_tokens,
        "total_s": round(total, 2),
        "out_tok_per_s": round(args.num_requests * args.max_tokens / total, 1),
        "ttft_p50_ms": round(pct(ttft, 0.5), 1),
        "ttft_p95_ms": round(pct(ttft, 0.95), 1),
        "ttft_p99_ms": round(pct(ttft, 0.99), 1),
        "ttft_max_ms": round(ttft[-1], 1),
        "e2e_p50_ms": round(pct(e2e, 0.5), 1),
        "e2e_p95_ms": round(pct(e2e, 0.95), 1),
    }
    print(json.dumps(out, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")           # 不传 = v1/v2 都跑
    ap.add_argument("--num-requests", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--arrive-interval", type=float, default=0.25)
    ap.add_argument("--gpu-mem-util", type=float, default=0.3)
    args = ap.parse_args()
    if args.engine:
        return worker(args)

    rows = []
    for e in (args.engines if hasattr(args, "engines") else ["v1", "v2"]):
        r = subprocess.run([sys.executable, __file__, "--engine", e,
                            "--num-requests", str(args.num_requests),
                            "--batch-size", str(args.batch_size),
                            "--max-tokens", str(args.max_tokens),
                            "--arrive-interval", str(args.arrive_interval),
                            "--gpu-mem-util", str(args.gpu_mem_util)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"{e} 崩溃: {r.stderr.strip().splitlines()[-1]}")
            continue
        rows.append(json.loads(r.stdout.strip().splitlines()[-1]))

    out_path = os.path.join(HERE, "results", "stress.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    for r in rows:
        print(f"===== {r['engine']}（{r['num_requests']} 请求 × 每 {r['arrive_interval_s']}s 到 1 个, "
              f"max_tokens={r['max_tokens']}） =====")
        print(f"  总时长 {r['total_s']}s, 输出吞吐 {r['out_tok_per_s']} tok/s")
        print(f"  TTFT p50/p95/p99/max = {r['ttft_p50_ms']}/{r['ttft_p95_ms']}/"
              f"{r['ttft_p99_ms']}/{r['ttft_max_ms']} ms")
        print(f"  E2E  p50/p95        = {r['e2e_p50_ms']}/{r['e2e_p95_ms']} ms")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
