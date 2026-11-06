"""Experiment orchestrator: one subprocess per parallel configuration.

Phase "sweep":  bench_workload_matrix.py per config (throughput/latency matrix).
Phase "trace":  capture_parallel_window.py per config (CUDA trace windows,
                single-DP entry only; DP>1 configs are skipped there).
Phase "all":    sweep then trace.

Post-3318af2 upstream aligns EP with DP x TP (vLLM coupled form):
  enable_expert_parallel => ep group = dp_size x tp_size (never a fifth axis).
Tag semantics follow the coupled mapping:
  tp8    = DP1 x TP8, experts sharded by TP (EP off)
  ep8    = DP1 x TP8 + EP  (EP group = 8)
  tp4ep2 = DP2 x TP4 + EP  (EP group = 8, attention TP4 per replica)
  tp2ep4 = DP4 x TP2 + EP  (EP group = 8, attention TP2 per replica)

Example (4 free GPUs):
  python tests/run_experiments.py --phase sweep \
      --model /path/to/Qwen3-Omni-30B-A3B-Instruct --gpus 2,3,4,7 \
      --configs tp4,ep4,tp2ep2 --out /path/to/results

Results layout under --out:
  sweep/<tag>/results.json + config.json + inputs.json + correctness.json
  sweep/<tag>/memory_rank*.json
  trace/<tag>_<workload>/{trace,trace/capture_rank*.json,config.json,outputs.json}
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# tag -> dict(tp, dp, pp, ep)      world_size = tp*dp*pp (EP overlays DP x TP)
CONFIGS = {
    "tp8": dict(tp=8, dp=1, pp=1, ep=False),
    "ep8": dict(tp=8, dp=1, pp=1, ep=True),
    "tp4ep2": dict(tp=4, dp=2, pp=1, ep=True),
    "tp2ep4": dict(tp=2, dp=4, pp=1, ep=True),
    "tp4": dict(tp=4, dp=1, pp=1, ep=False),
    "ep4": dict(tp=4, dp=1, pp=1, ep=True),
    "tp2ep2": dict(tp=2, dp=2, pp=1, ep=True),
    "pp2tp2ep2": dict(tp=2, dp=2, pp=2, ep=True),
}

# capture_parallel_window 每条请求 max_tokens=4 → 捕获 run 的总消息数
# M_capture = M_sweep - max_out_sweep + 4（调度步数与并行配置无关，
# 因为 budget/max_num_seqs 相同）。decode 净窗口 = 捕获 run 末尾 4 条消息。
TRACE_SPECS = [
    # (workload, requests, sweep_out, capture_decode_window, capture_prefill_window)
    ("mixed64", 64, 64, 4, 0),
    ("short64", 64, 32, 4, 0),
    ("context4k8", 8, 32, 4, 8),
]
TRACE_SPECS_DECODE_ONLY = True


def trace_windows_from_sweep(sweep_result_json):
    """读 sweep 结果的 step 数 → 推导捕获总消息数与窗口。

    返回 [(job_tag, workload, requests, start_message, messages)]。
    """
    with open(sweep_result_json, encoding="utf-8") as f:
        records = {r["workload"]: r for r in json.load(f)
                   if r.get("repeat", 0) >= 0}
    jobs = []
    for wl, n_req, sweep_out, dec_msgs, pre_msgs in TRACE_SPECS:
        r = records.get(wl)
        if not r:
            continue
        s = len(r["step_s"])
        m = s - sweep_out + 4                    # 捕获 run 总消息数
        dec_start = max(2, m - dec_msgs + 1)     # 末尾 dec_msgs 条
        jobs.append((f"{wl}_decode", wl, n_req, dec_start, dec_msgs))
        if pre_msgs > 0 and dec_start > 4:       # 前段 prefill 窗口
            jobs.append((f"{wl}_prefill", wl, n_req, 2,
                         min(pre_msgs, dec_start - 2)))
    return jobs


def run_one(cmd, tag, out_dir, env_extra, timeout):
    log = os.path.join(out_dir, f"log_{tag}.txt")
    env = dict(os.environ)
    env.update(env_extra)
    t0 = time.perf_counter()
    print(f"[orch] start {tag}", flush=True)
    try:
        with open(log, "w", encoding="utf-8") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                                   timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        code = -9
        print(f"[orch] TIMEOUT {tag} (killed)", flush=True)
    dt = (time.perf_counter() - t0) / 60
    print(f"[orch] {'OK' if code == 0 else 'FAIL'} {tag} exit={code} "
          f"{dt:.1f}min log={log}", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["sweep", "trace", "all"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", required=True, help="comma list, e.g. 2,3,4,7")
    ap.add_argument("--configs", default="tp8,ep8,tp4ep2,tp2ep4")
    ap.add_argument("--workloads",
                    default="short64,short128,context1k8,context4k8,mixed64,arrival64")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--expert-backend", default=None)
    ap.add_argument("--timeout", type=int, default=14400)
    args = ap.parse_args()

    gpus = args.gpus
    n_gpu = len(gpus.split(","))
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    tags = [t.strip() for t in args.configs.split(",") if t.strip()]
    for tag in tags:
        if tag not in CONFIGS:
            raise SystemExit(f"unknown config {tag}")
        c = CONFIGS[tag]
        if c["tp"] * c["dp"] * c["pp"] != n_gpu:
            raise SystemExit(f"{tag}: tp*dp*pp={c['tp']*c['dp']*c['pp']} "
                             f"!= {n_gpu} gpus")

    py = sys.executable
    sweep_dir = None
    if args.phase in ("sweep", "all"):
        for i, tag in enumerate(tags):
            c = CONFIGS[tag]
            cmd = [py, os.path.join(HERE, "bench_workload_matrix.py"),
                   "--model", args.model,
                   "--output", os.path.join(out, "sweep", tag + args.tag_suffix),
                   "--tp", str(c["tp"]), "--pp", str(c["pp"]),
                   "--dp", str(c["dp"]),
                   "--gpus", gpus, "--master-port", str(34100 + i * 17),
                   "--workloads", args.workloads,
                   "--repeats", str(args.repeats)]
            if c["ep"]:
                cmd.append("--enable-expert-parallel")
            if args.expert_backend:
                cmd += ["--expert-backend", args.expert_backend]
            code = run_one(cmd, f"sweep_{tag}{args.tag_suffix}", out, {},
                           args.timeout)
            if code != 0:
                print(f"[orch] abort on failed sweep {tag}", flush=True)
                sys.exit(1)
            if sweep_dir is None:
                sweep_dir = os.path.join(out, "sweep", tag + args.tag_suffix)

    if args.phase in ("trace", "all"):
        # 捕获窗口从 sweep 的 step 数推导（scheduler 步数与并行配置无关）；
        # 找不到 sweep 结果时用保守回退窗口。
        jobs = None
        probe = sweep_dir or (os.path.join(out, "sweep", tags[0] + args.tag_suffix))
        rj = os.path.join(probe, "results.json")
        if os.path.exists(rj):
            jobs = trace_windows_from_sweep(rj)
        else:
            jobs = [("mixed64_decode", "mixed64", 64, 60, 4),
                    ("short64_decode", "short64", 64, 8, 4)]
        print(f"[orch] trace jobs: {jobs}", flush=True)
        for i, tag in enumerate(tags):
            c = CONFIGS[tag]
            if c["dp"] > 1:
                print(f"[orch] skip trace for {tag}: capture entry is "
                      f"single-DP", flush=True)
                continue
            for j, (job, wl, n_req, start_msg, n_msgs) in enumerate(jobs):
                cmd = [py, os.path.join(HERE, "capture_parallel_window.py"),
                       "--model", args.model,
                       "--inputs", os.path.join(probe, "inputs.json"),
                       "--output", os.path.join(out, "trace",
                                                f"{tag}_{job}{args.tag_suffix}"),
                       "--tp", str(c["tp"]), "--pp", str(c["pp"]),
                       "--gpus", gpus, "--master-port",
                       str(35100 + i * 29 + j),
                       "--workload", wl, "--requests", str(n_req),
                       "--start-message", str(start_msg),
                       "--messages", str(n_msgs)]
                if c["ep"]:
                    cmd.append("--enable-expert-parallel")
                env_extra = {}
                if args.expert_backend:
                    env_extra["VLLM_LEARN_EXPERT_BACKEND"] = args.expert_backend
                code = run_one(cmd, f"trace_{tag}_{job}{args.tag_suffix}",
                               out, env_extra, args.timeout)
                if code != 0:
                    print(f"[orch] trace fail {tag}_{job} (continue)",
                          flush=True)
        print("[orch] done", flush=True)
        return

    print("[orch] done", flush=True)

if __name__ == "__main__":
    main()
