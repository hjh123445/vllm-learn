# benchmark/profile_pipeline.py
"""推理链路 profile：torch.profiler 包住完整生成循环。
- 阶段计时（prefill / decode / mask / sample）
- 按 CUDA 时间排序的 top 算子表
- Chrome trace（chrome://tracing 打开看 kernel 时间线与 GPU 空闲）

用法:
  python benchmark/profile_pipeline.py --engine v1 --num-requests 8 --max-tokens 32
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["v0", "v1"], required=True)
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    out_file = os.path.join(HERE, "results", f"profile_{args.engine}.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    subprocess.run([sys.executable, os.path.join(HERE, "harness.py"),
                    "--engine", args.engine,
                    "--num-requests", str(args.num_requests),
                    "--batch-size", str(args.batch_size),
                    "--max-tokens", str(args.max_tokens),
                    "--profile", "--out", out_file],
                   check=True, capture_output=True)

    with open(out_file) as f:
        d = json.load(f)
    st = d["stage_times"]
    print(f"===== profile {args.engine}（{args.num_requests} 请求 × {args.max_tokens} tok） =====")
    print("阶段耗时:")
    for k in sorted(st, key=st.get, reverse=True):
        print(f"  {k:8s} {st[k]:7.2f} s")
    print(f"\ntop 算子（按 CUDA 时间）:\n")
    print(d["profiler_table"])
    print(f"\nChrome trace: {d['trace']}")
    print("浏览器打开 chrome://tracing → Load，加载上面的 trace 文件；")
    print("GPU 行 kernel 之间的空白 = GPU 空闲（等 CPU），是下一步优化目标。")


if __name__ == "__main__":
    main()
