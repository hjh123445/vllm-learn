"""Aggregate experiment results -> p50/p90/p99 latency, throughput,
TTFT/TPOT, plus CUDA-trace breakdown and rank memory.

  python tests/analyze_results.py --dir <results_root>

Writes <dir>/analysis.json and <dir>/SUMMARY.md.
TTOT definition used here: per-request mean inter-decode-token latency
((last_arrival - first_arrival)/(n_decode-1)); 同时汇总全量 ITL 样本分位数。
"""
import argparse
import glob
import json
import os
import re
import statistics

DECODE_MSG_RE = re.compile(r"rank(\d+)\.stage(\d+)\.message(\d+)\.mb(\d+)\.decode")
PREFILL_MSG_RE = re.compile(r"rank(\d+)\.stage(\d+)\.message(\d+)\.mb(\d+)\.prefill")
ATTN_RE = re.compile(r"stage\d+\.layer\d+\.attention")
MOE_RE = re.compile(r"stage\d+\.layer\d+\.moe")


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    f = pos - lo
    return xs[lo] * (1 - f) + xs[hi] * f


def ms(v):
    return None if v is None else round(v * 1e3, 2)


def block(name, xs):
    return {f"{name}_p50_ms": ms(pct(xs, 50)), f"{name}_p90_ms": ms(pct(xs, 90)),
            f"{name}_p99_ms": ms(pct(xs, 99)),
            f"{name}_mean_ms": ms(statistics.mean(xs) if xs else None),
            f"{name}_n": len(xs)}


def analyze_sweep(results_json):
    with open(results_json, encoding="utf-8") as f:
        records = json.load(f)
    by_wl = {}
    for r in records:
        if r.get("repeat", 0) < 0:           # warmup 排除
            continue
        by_wl.setdefault(r["workload"], []).append(r)
    out = {}
    for wl, recs in by_wl.items():
        req = [q for r in recs for q in r["requests"]]
        ttft = [q["ttft_s"] for q in req]
        e2e = [q["e2e_s"] for q in req]
        tpot = [q["tpot_s"] for q in req if q.get("tpot_s") is not None]
        itls = [b - a for q in req for a, b in
                zip(q["arrival_s"], q["arrival_s"][1:])]
        prompt_toks = sum(q["prompt_tokens"] for q in req)
        out_toks = sum(q["output_tokens"] for q in req)
        tps = [r["output_tok_s"] for r in recs]
        steps = [s for r in recs for s in r["step_s"]]
        row = {"workload": wl, "repeats": len(recs), "requests": len(req),
               "prompt_tokens": prompt_toks, "output_tokens": out_toks,
               "wall_s_mean": round(statistics.mean(r["wall_s"] for r in recs), 2),
               "output_tps_median": round(statistics.median(tps), 2),
               "output_tps_min": round(min(tps), 2),
               "output_tps_max": round(max(tps), 2),
               }
        row["total_tps_median"] = round(
            (prompt_toks + out_toks) / len(recs) /
            statistics.mean(r["wall_s"] for r in recs), 2)
        row.update(block("ttft", ttft))
        row.update(block("e2e", e2e))
        row.update(block("ttot", tpot))      # 逐请求平均 TPOT
        row.update(block("itl", itls))       # 全量逐 token 间隔
        row.update(block("step", steps))
        out[wl] = row
    return out


def classify_kernel(name):
    n = name.lower()
    if "nccl" in n:
        if "allreduce" in n:
            return "nccl_allreduce"
        if "allgather" in n:
            return "nccl_allgather"
        if "alltoall" in n:
            return "nccl_alltoall"
        if "reducescatter" in n:
            return "nccl_reducescatter"
        if "send" in n or "recv" in n:
            return "nccl_p2p"
        return "nccl_other"
    for k in ("gemm", "sgemm", "hgemm", "nvjet", "cutlass", "matmul"):
        if k in n:
            return "gemm"
    if ("elementwise" in n or "vectorized" in n or "index" in n
            or "copy" in n or "catarray" in n or "gather" in n
            or "scatter" in n or "reduce_" in n):
        return "elementwise_copy"
    if "triton" in n or "fa_" in n or "attn" in n or "fused" in n:
        return "triton_attn_fused"
    return "other"


def analyze_trace_dir(tdir):
    files = sorted(glob.glob(os.path.join(tdir, "cuda_rank*.json")))
    if not files:
        return None
    ranks = []
    for p in files:
        m = re.search(r"cuda_rank(\d+)", os.path.basename(p))
        with open(p, encoding="utf-8") as f:
            evs = json.load(f).get("traceEvents", [])
        kern = [(e["name"], e.get("ts", 0), e.get("dur", 0)) for e in evs
                if e.get("ph") == "X" and e.get("cat") in
                ("kernel", "gpu_memcpy", "gpu_memset")]
        ann = [(e["name"], e.get("dur", 0)) for e in evs
               if e.get("ph") == "X" and e.get("cat") == "user_annotation"]
        if not kern:
            continue
        ts0 = min(t for _, t, _ in kern)
        ts1 = max(t + d for _, t, d in kern)
        wall = ts1 - ts0
        busy = sum(d for _, _, d in kern)
        byc = {}
        kagg = {}
        for name, _, dur in kern:
            c = classify_kernel(name)
            byc[c] = byc.get(c, 0) + dur
            a = kagg.setdefault(name, [0, 0])
            a[0] += dur
            a[1] += 1
        annsum = {}
        msg_steps = {}
        for name, dur in ann:
            if DECODE_MSG_RE.match(name):
                msg_steps.setdefault("decode", []).append(dur)
            elif PREFILL_MSG_RE.match(name):
                msg_steps.setdefault("prefill", []).append(dur)
            elif ATTN_RE.match(name):
                annsum["attention"] = annsum.get("attention", 0) + dur
            elif MOE_RE.match(name):
                annsum["moe"] = annsum.get("moe", 0) + dur
            elif name.startswith("moe."):
                annsum[name] = annsum.get(name, 0) + dur
        top = sorted(kagg.items(), key=lambda kv: -kv[1][0])[:12]
        nccl = sum(v for k, v in byc.items() if k.startswith("nccl"))
        ranks.append({
            "rank": int(m.group(1)) if m else -1,
            "file_mb": round(os.path.getsize(p) / 1e6, 1),
            "wall_ms": round(wall / 1e3, 2),
            "gpu_busy_ms": round(busy / 1e3, 2),
            "gpu_busy_pct": round(100 * busy / max(1, wall), 1),
            "nccl_ms": round(nccl / 1e3, 2),
            "nccl_busy_pct": round(100 * nccl / max(1, busy), 1),
            "kernel_class_ms": {k: round(v / 1e3, 2) for k, v in
                                sorted(byc.items(), key=lambda kv: -kv[1])},
            "top_kernels": [{"name": n[:80], "ms": round(a[0] / 1e3, 2),
                             "n": a[1]} for n, a in top],
            "annot_ms": {k: round(v / 1e3, 2) for k, v in annsum.items()},
            "decode_msgs": len(msg_steps.get("decode", [])),
            "decode_msg_ms_mean": ms(statistics.mean(msg_steps["decode"]))
            if msg_steps.get("decode") else None,
            "prefill_msgs": len(msg_steps.get("prefill", [])),
            "prefill_msg_ms_mean": ms(statistics.mean(msg_steps["prefill"]))
            if msg_steps.get("prefill") else None,
        })
    out = {"ranks": ranks}
    dec = [r["decode_msg_ms_mean"] for r in ranks if r["decode_msg_ms_mean"]]
    if len(dec) > 1:
        out["decode_rank_imbalance"] = round(max(dec) / max(1e-9, min(dec)), 2)
    return out


def analyze_memory(metrics_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(metrics_dir, "memory_rank*.json"))):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        return None
    gb = 1024 ** 3
    return {
        "n_ranks": len(rows),
        "param_gb_per_rank": [round(r["local_parameter_bytes"] / gb, 2)
                              for r in rows],
        "kv_gb_per_rank": [round(r["kv_cache_bytes"] / gb, 2) for r in rows],
        "peak_alloc_gb_per_rank": [round(r["peak_allocated_bytes"] / gb, 2)
                                   for r in rows],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    root = args.dir
    out = {"root": os.path.abspath(root), "sweeps": {}, "traces": {},
           "memory": {}}
    for p in sorted(glob.glob(os.path.join(root, "sweep", "*", "results.json"))):
        tag = p.split(os.sep)[-2]
        out["sweeps"][tag] = analyze_sweep(p)
    for p in sorted(glob.glob(os.path.join(root, "sweep", "*"))):
        if os.path.isdir(p):
            m = analyze_memory(p)
            if m:
                out["memory"][os.path.basename(p)] = m
    for td in sorted(glob.glob(os.path.join(root, "trace", "*"))):
        t = analyze_trace_dir(os.path.join(td, "trace"))
        if t:
            out["traces"][os.path.basename(td)] = t
    with open(os.path.join(root, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- markdown ----
    L = ["# 实验汇总\n",
         "## 吞吐与时延（p50/p90/p99，重复 runs 汇总）", ""]
    cols = ["config", "workload", "req", "out_tps_med", "tot_tps_med",
            "ttft_p50", "ttft_p90", "ttft_p99", "ttot_p50", "ttot_p90",
            "ttot_p99", "e2e_p50", "e2e_p90", "e2e_p99"]
    L.append("|" + "|".join(cols) + "|")
    L.append("|" + "---|" * len(cols))
    for tag, sweeps in out["sweeps"].items():
        for wl, r in sweeps.items():
            L.append("|" + "|".join(str(x) for x in (
                tag, wl, r["requests"], r["output_tps_median"],
                r["total_tps_median"], r["ttft_p50_ms"], r["ttft_p90_ms"],
                r["ttft_p99_ms"], r["ttot_p50_ms"], r["ttot_p90_ms"],
                r["ttot_p99_ms"], r["e2e_p50_ms"], r["e2e_p90_ms"],
                r["e2e_p99_ms"])) + "|")
    L += ["", "## 每 rank 显存（GB）", ""]
    for tag, m in out["memory"].items():
        L.append(f"- **{tag}**: param={m['param_gb_per_rank']} "
                 f"kv={ [round(v,1) for v in m['kv_gb_per_rank']] } "
                 f"peak={m['peak_alloc_gb_per_rank']}")
    L += ["", "## Trace 分解（每 trace 目录汇总 rank0-视角及全 rank 极值）", ""]
    for name, t in out["traces"].items():
        if not t.get("ranks"):
            continue
        r0 = next((r for r in t["ranks"] if r["rank"] == 0), t["ranks"][0])
        L.append(f"### {name}")
        L.append(f"- decode 窗口消息数 rank0={r0['decode_msgs']} "
                 f"均 {r0['decode_msg_ms_mean']}ms; 全 rank decode 均值倾斜="
                 f"{t.get('decode_rank_imbalance')}")
        L.append(f"- rank0: wall={r0['wall_ms']}ms busy={r0['gpu_busy_ms']}ms "
                 f"({r0['gpu_busy_pct']}%) nccl={r0['nccl_ms']}ms; "
                 f"kernel 分类={r0['kernel_class_ms']}")
        L.append(f"- 注解 decomposition={r0['annot_ms']}")
        L.append(f"- top kernels: " + "; ".join(
            f"{k['name'][:55]} {k['ms']}ms×{k['n']}" for k in r0['top_kernels'][:6]))
        L.append("")
    with open(os.path.join(root, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[analyze] wrote", os.path.join(root, "analysis.json"),
          "and SUMMARY.md")


if __name__ == "__main__":
    main()
