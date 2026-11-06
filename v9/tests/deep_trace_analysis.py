"""Deep per-step trace analysis:
1. comm/compute overlap on the timeline (NCCL stream vs compute stream)
2. per-decode-step hot breakdown (message-annotated windows)
3. small-op census -> fusion candidates (kernel count, total us, gaps)
4. host sync points (DtoH memcpy, cuda*Synchronize) and aten op counts

Usage: python deep_trace_analysis.py --trace-dir <dir with cuda_rank*.json>
       [--ranks 0,4] --out <out.json>
All times in the report are microseconds unless suffixed _ms.
"""
import argparse
import collections
import glob
import json
import os
import re

MSG_RE = re.compile(r"rank(\d+)\.stage(\d+)\.message(\d+)\.mb(\d+)\.(decode|prefill)")
MB_RE = re.compile(r"stage(\d+)\.microbatch(\d+)\.lane(\d+)\.(decode|prefill)")

FUSION_HINTS = {
    "elementwise": "pointwise (add/mul/silu/norm cast) - fusable",
    "index_elementwise": "gather/index & scatter - fusable into dispatch/combine",
    "vectorized_elementwise": "pointwise - fusable",
    "unrolled_elementwise": "small pointwise - fusable",
    "reduce_kernel": "reduction (softmax/sum/topk parts) - fusable",
    "Memcpy DtoD": "layout/copy - removable or fusable",
    "catArrayBatchedCopy": "torch.cat - removable by preallocated buffer",
    "splitKreduce": "gemm splitK epilogue - inherent to small gemm",
}


def union_len(ivs):
    """合并 [s,e) 区间求并集长度。"""
    if not ivs:
        return 0
    ivs = sorted(ivs)
    total = 0
    s, e = ivs[0]
    for a, b in ivs[1:]:
        if a > e:
            total += e - s
            s, e = a, b
        else:
            e = max(e, b)
    return total + e - s


def intersect_len(ivs_a, ivs_b):
    a, b = sorted(ivs_a), sorted(ivs_b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def classify(name):
    n = name.lower()
    if "nccl" in n:
        if "allreduce" in n:
            return "nccl_allreduce"
        if "alltoall" in n:
            return "nccl_alltoall"
        if "allgather" in n:
            return "nccl_allgather"
        if "sendrecv" in n or "send" in n or "recv" in n:
            return "nccl_sendrecv"
        return "nccl_other"
    if any(k in n for k in ("gemm", "nvjet", "cutlass_80_tensorop",
                            "s16816gemm", "splitkreduce", "wmma")):
        return "gemm"
    if any(k in n for k in ("flash_fwd", "fmha_", "fa_", "attn")):
        return "attention_kernel"
    if "memcpy" in n or "memset" in n:
        return "memcpy"
    for k in ("elementwise", "index", "catarray", "reduce_kernel",
              "scatter", "gather", "copy"):
        if k in n:
            return "small_pointwise"
    if "triton" in n:
        return "triton"
    return "other"


class RankTrace:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            evs = json.load(f).get("traceEvents", [])
        self.kernels = []       # (name, ts, dur, tid)
        self.cpu_ops = []       # (name, ts, dur)
        self.runtime = []       # (name, ts, dur)
        self.annotations = []   # (name, ts, dur)
        self.memcpys = []       # (name, ts, dur, tid)
        for e in evs:
            if e.get("ph") != "X":
                continue
            cat = e.get("cat", "")
            tup = (e.get("name", ""), e.get("ts", 0), e.get("dur", 0),
                   e.get("tid", 0))
            if cat == "kernel":
                self.kernels.append(tup)
            elif cat == "cpu_op":
                self.cpu_ops.append(tup)
            elif cat in ("cuda_runtime", "cuda_driver"):
                self.runtime.append(tup)
            elif cat == "user_annotation":
                self.annotations.append(tup)
            elif cat in ("gpu_memcpy", "gpu_memset"):
                self.memcpys.append(tup)
        self.kernels.sort(key=lambda x: x[1])
        self.annotations.sort(key=lambda x: x[1])


def step_windows(rt):
    """用 message 注解的起始时间框出每个 step 的窗口（dur 有异常值，不用）。"""
    msgs = []
    for name, ts, dur, _ in rt.annotations:
        m = MSG_RE.match(name)
        if m:
            msgs.append((ts, m.group(5), int(m.group(3))))
    if not msgs:
        return []
    msgs.sort()
    ts_min = msgs[0][0]
    ts_max = max(t + d for _, t, d, _ in rt.kernels) if rt.kernels \
        else msgs[-1][0] + 1
    wins = []
    for i, (ts, kind, mid) in enumerate(msgs):
        end = msgs[i + 1][0] if i + 1 < len(msgs) else ts_max
        wins.append((ts, end, kind, mid))
    return wins


def analyze_window(rt, ts0, ts1):
    kerns = [(n, t, d, tid) for n, t, d, tid in rt.kernels
             if ts0 <= t < ts1]
    if not kerns:
        return None
    wall = ts1 - ts0
    byc = collections.defaultdict(int)
    cnt = collections.defaultdict(int)
    nccl_ivs, comp_ivs = [], []
    by_tid = collections.defaultdict(list)
    kname_agg = collections.defaultdict(lambda: [0, 0, 0])  # dur, cnt, max
    small_agg = collections.defaultdict(lambda: [0, 0])     # dur, cnt (<20us)
    for n, t, d, tid in kerns:
        c = classify(n)
        byc[c] += d
        cnt[c] += 1
        iv = (t, t + d)
        if c.startswith("nccl"):
            nccl_ivs.append(iv)
        elif c != "memcpy":
            comp_ivs.append(iv)
        by_tid[tid].append((t, t + d))
        a = kname_agg[n]
        a[0] += d
        a[1] += 1
        a[2] = max(a[2], d)
        if d < 20:
            sa = small_agg[n]
            sa[0] += d
            sa[1] += 1
    busy = union_len([(t, t + d) for _, t, d, _ in kerns])
    nccl_u = union_len(nccl_ivs)
    comp_u = union_len(comp_ivs)
    overlap = intersect_len(nccl_ivs, comp_ivs)
    # 主计算 stream 上的间隙（kernel 间空隙）
    tid_main = max(by_tid, key=lambda k: len(by_tid[k])) if by_tid else None
    gaps = []
    if tid_main is not None:
        ivs = sorted(by_tid[tid_main])
        for (a0, a1), (b0, b1) in zip(ivs, ivs[1:]):
            g = b0 - a1
            if g > 0:
                gaps.append(g)
    gaps.sort()
    top = sorted(kname_agg.items(), key=lambda kv: -kv[1][0])[:15]
    small_top = sorted(small_agg.items(), key=lambda kv: -kv[1][0])[:15]
    return {
        "wall_us": wall,
        "busy_us": busy,
        "busy_pct": round(100 * busy / max(1, wall), 1),
        "by_class_us": dict(sorted(byc.items(), key=lambda kv: -kv[1])),
        "by_class_cnt": dict(cnt),
        "nccl_union_us": nccl_u,
        "comp_union_us": comp_u,
        "comm_comp_overlap_us": overlap,
        "comm_overlap_pct": round(100 * overlap / max(1, nccl_u), 1),
        "main_stream_gaps": {
            "n": len(gaps),
            "sum_us": sum(gaps),
            "sum_gt20us": sum(g for g in gaps if g > 20),
            "n_gt20us": sum(1 for g in gaps if g > 20),
            "sum_gt200us": sum(g for g in gaps if g > 200),
            "n_gt200us": sum(1 for g in gaps if g > 200),
            "p50_us": gaps[len(gaps) // 2] if gaps else 0,
            "p90_us": gaps[int(len(gaps) * .9)] if gaps else 0,
        },
        "top_kernels": [
            {"name": n[:70], "total_us": a[0], "n": a[1], "max_us": a[2],
             "avg_us": round(a[0] / a[1], 1)}
            for n, a in top],
        "small_kernels_lt20us": [
            {"name": n[:70], "total_us": a[0], "n": a[1],
             "avg_us": round(a[0] / a[1], 1)}
            for n, a in small_top],
    }


def global_stats(rt):
    """全文件级：同步调用、DtoH、aten op、annotation 层级耗时。"""
    sync = collections.Counter()
    for n, _, _, _ in rt.runtime:
        for k in ("cudaDeviceSynchronize", "cudaStreamSynchronize",
                  "cudaEventSynchronize", "cudaMemcpyAsync", "cudaMemcpy",
                  "cudaEventRecord", "cudaStreamWaitEvent", "cudaLaunchKernel"):
            if k in n:
                sync[k] += 1
                break
    memcpy = collections.Counter()
    memcpy_us = collections.Counter()
    for n, _, d, _ in rt.memcpys:
        k = "DtoH" if "DtoH" in n else ("HtoD" if "HtoD" in n else
                                        ("DtoD" if "DtoD" in n else n[:20]))
        memcpy[k] += 1
        memcpy_us[k] += d
    aten = collections.Counter()
    aten_us = collections.Counter()
    for n, _, d, _ in rt.cpu_ops:
        aten[n] += 1
        aten_us[n] += d
    top_aten = sorted(aten_us.items(), key=lambda kv: -kv[1])[:20]
    ann = collections.Counter()
    for n, _, d, _ in rt.annotations:
        key = n if (n.startswith("moe.") or "attention" in n or MB_RE.match(n)) \
            else re.sub(r"\d+", "N", n)[:60]
        ann[key] += d
    top_ann = sorted(ann.items(), key=lambda kv: -kv[1])[:18]
    return {
        "sync_calls": dict(sync),
        "memcpy_calls": dict(memcpy),
        "memcpy_us": dict(memcpy_us),
        "top_aten_by_us": [(n[:60], us, aten[n]) for n, us in top_aten],
        "top_annotations_by_us": [(n[:60], us) for n, us in top_ann],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--ranks", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"dir": args.trace_dir, "ranks": {}}
    for r in [int(x) for x in args.ranks.split(",")]:
        files = sorted(glob.glob(os.path.join(args.trace_dir,
                                              f"cuda_rank{r}*.json")))
        if not files:
            continue
        rt = RankTrace(files[0])
        rep = {"file_mb": round(os.path.getsize(files[0]) / 1e6, 1),
               "n_kernels": len(rt.kernels),
               "steps": [], "global": global_stats(rt)}
        for ts0, ts1, kind, mid in step_windows(rt):
            w = analyze_window(rt, ts0, ts1)
            if w:
                w.update({"kind": kind, "message": mid,
                          "dur_ms": round((ts1 - ts0) / 1e3, 1)})
                rep["steps"].append(w)
        # 全窗口重叠（不依赖注解）
        if rt.kernels:
            t0 = min(t for _, t, _, _ in rt.kernels)
            t1 = max(t + d for _, t, d, _ in rt.kernels)
            rep["whole_window"] = analyze_window(rt, t0, t1)
        out["ranks"][str(r)] = rep
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[deep] wrote", args.out)
    for r, rep in out["ranks"].items():
        ww = rep.get("whole_window")
        if ww:
            print(f"  rank{r}: kernels={rep['n_kernels']} "
                  f"busy={ww['busy_pct']}% "
                  f"comm_overlap={ww['comm_overlap_pct']}% "
                  f"gaps>200us={ww['main_stream_gaps']['n_gt200us']}")
        for s in rep["steps"]:
            print(f"    {s['kind']} msg{s['message']}: {s['dur_ms']}ms "
                  f"busy={s['busy_pct']}% nccl={s['by_class_us'].get('nccl_sendrecv',0)+s['by_class_us'].get('nccl_allreduce',0)}us "
                  f"overlap={s['comm_overlap_pct']}%")


if __name__ == "__main__":
    main()
