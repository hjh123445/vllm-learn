# benchmark/speed.py
"""速度测试：TTFT / TPOT / E2E / 吞吐。通过子进程调 harness worker，避免 v0/v1 模块冲突。
用法:
  python benchmark/speed.py --engine v0 --num-requests 12 --batch-size 8
  python benchmark/speed.py --engine v1 --num-requests 12 --batch-size 8
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(engine, num_requests, batch_size, max_tokens=None):
    """子进程调 harness worker，JSON 走临时文件（v2 初始化时会打显存预算报告，stdout 不干净）。"""
    import tempfile
    out_file = os.path.join(HERE, "results", f"harness_{engine}.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "harness.py"),
           "--engine", engine, "--num-requests", str(num_requests),
           "--batch-size", str(batch_size), "--out", out_file]
    if max_tokens:
        cmd += ["--max-tokens", str(max_tokens)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    with open(out_file) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["v0", "v1", "v2"], required=True)
    ap.add_argument("--num-requests", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="统一 max_tokens（默认用 32/96/160 混合负载）")
    args = ap.parse_args()

    d = run(args.engine, args.num_requests, args.batch_size, args.max_tokens)
    reqs = d["requests"]
    ttft = [r["first_token_t"] for r in reqs]
    tpot = [(r["finish_t"] - r["first_token_t"]) / max(r["n_tokens"] - 1, 1)
            for r in reqs]
    e2e = [r["finish_t"] for r in reqs]
    total_out = sum(r["n_tokens"] for r in reqs)

    print(f"===== {args.engine}  ({args.num_requests} 请求, batch_size={args.batch_size}, "
          f"max_tokens 负载: {'统一 ' + str(args.max_tokens) if args.max_tokens else '32/96/160 混合'}) =====")
    print(f"TTFT   首 token 时延   avg {sum(ttft)/len(ttft)*1e3:8.1f} ms   "
          f"p50 {sorted(ttft)[len(ttft)//2]*1e3:8.1f} ms   max {max(ttft)*1e3:8.1f} ms")
    print(f"TPOT   每 token 时延   avg {sum(tpot)/len(tpot)*1e3:8.1f} ms   "
          f"max {max(tpot)*1e3:8.1f} ms")
    print(f"E2E    完成时延       avg {sum(e2e)/len(e2e)*1e3:8.1f} ms   "
          f"p50 {sorted(e2e)[len(e2e)//2]*1e3:8.1f} ms   max {max(e2e)*1e3:8.1f} ms")
    print(f"吞吐   输出 {total_out} tok / {d['total_time']:.2f} s = "
          f"{total_out/d['total_time']:.1f} tok/s,  {len(reqs)/d['total_time']:.2f} req/s")
    st = d["stage_times"]
    print("阶段   " + "  ".join(f"{k} {v:.2f}s" for k, v in sorted(st.items())))

    out_path = os.path.join(HERE, "results", f"speed_{args.engine}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": d["meta"], "ttft": ttft, "tpot": tpot, "e2e": e2e,
                   "total_time": d["total_time"], "total_out_tokens": total_out,
                   "stage_times": st}, f, ensure_ascii=False, indent=1)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
