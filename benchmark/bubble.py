# benchmark/bubble.py
"""流水线气泡可视化：同样的负载跑 v0（静态批处理）和 v1（迭代级调度），
画请求时间线 Gantt 图 + batch 利用率曲线，量化静态批处理的气泡浪费。

用法:
  python benchmark/bubble.py --num-requests 12 --batch-size 8
输出:
  results/bubble_v0_vs_v1.png, results/bubble_summary.json
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_harness(engine, num_requests, batch_size):
    out_file = os.path.join(HERE, "results", f"bubble_{engine}.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    subprocess.run([sys.executable, os.path.join(HERE, "harness.py"),
                   "--engine", engine, "--num-requests", str(num_requests),
                   "--batch-size", str(batch_size), "--out", out_file],
                  check=True, capture_output=True)
    with open(out_file) as f:
        return json.load(f)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-requests", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    v0 = run_harness("v0", args.num_requests, args.batch_size)
    v1 = run_harness("v1", args.num_requests, args.batch_size)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                             gridspec_kw={"height_ratios": [4, 4, 1.6]})
    fig.suptitle(f"Static batching (v0) vs Iteration-level scheduling (v1)  "
                 f"({args.num_requests} requests, batch_size={args.batch_size})",
                 fontsize=13)

    # ---- v0 Gantt ----
    ax = axes[0]
    for i, r in enumerate(v0["requests"]):
        bs, be, ft = r["batch_start"], r["batch_end"], r["finish_t"]
        ax.barh(i, ft - bs, left=bs, height=0.7, color="#4C72B0")          # 有效生成
        ax.barh(i, be - ft, left=ft, height=0.7, color="#C44E52",          # 气泡：完成后仍占位
                hatch="//", edgecolor="#8C1C1F", linewidth=0.3)
    ax.set_title("v0 static batching: red hatched = bubble (request finished, slot still occupied)")
    ax.set_ylabel("request")

    # ---- v1 Gantt ----
    ax = axes[1]
    for i, r in enumerate(v1["requests"]):
        adm, ft = r["admit_t"], r["finish_t"]
        ax.barh(i, adm, left=0, height=0.7, color="#BBBBBB")               # 等待准入
        ax.barh(i, ft - adm, left=adm, height=0.7, color="#55A868")        # 有效生成
    ax.set_title("v1 iteration-level scheduling: green = active, gray = waiting to be admitted")
    ax.set_ylabel("request")

    for ax in axes[:2]:
        ax.set_xlim(0, max(v0["total_time"], v1["total_time"]))
        ax.grid(axis="x", alpha=0.3)

    # ---- 利用率曲线 ----
    ax = axes[2]
    for d, color, label in ((v0, "#4C72B0", "v0 active/batch"),
                            (v1, "#55A868", "v1 running/batch")):
        ts = [s["t"] for s in d["steps"]]
        util = [s["n_active"] / args.batch_size for s in d["steps"]]
        ax.plot(ts, util, color=color, label=label, linewidth=1.2)
    ax.set_title("batch utilization over time (1.0 = all slots generating useful tokens)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("utilization")
    ax.legend()
    ax.grid(alpha=0.3)

    legend = [Patch(color="#4C72B0", label="useful generation"),
              Patch(facecolor="#C44E52", hatch="//", edgecolor="#8C1C1F",
                    label="bubble: slot wasted (v0)"),
              Patch(color="#BBBBBB", label="waiting to be admitted (v1)")]
    axes[0].legend(handles=legend, loc="lower right", fontsize=8)

    # ---- 汇总数字 ----
    v0_useful = sum(r["finish_t"] - r["batch_start"] for r in v0["requests"])
    v0_occupy = sum(r["batch_end"] - r["batch_start"] for r in v0["requests"])
    v1_useful = sum(r["finish_t"] - r["admit_t"] for r in v1["requests"])
    v1_wall = sum(r["finish_t"] for r in v1["requests"])  # v1 全部请求并跑，占位≈活跃
    summary = {
        "v0": {"total_time": v0["total_time"],
               "useful_slot_time": round(v0_useful, 2),
               "occupied_slot_time": round(v0_occupy, 2),
               "bubble_ratio": round(1 - v0_useful / v0_occupy, 4)},
        "v1": {"total_time": v1["total_time"],
               "useful_slot_time": round(v1_useful, 2)},
        "speedup_total_wall_time": round(v0["total_time"] / v1["total_time"], 3),
    }
    out_png = os.path.join(HERE, "results", "bubble_v0_vs_v1.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    out_json = os.path.join(HERE, "results", "bubble_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"saved to {out_png} / {out_json}")


if __name__ == "__main__":
    main()
