"""图编译裁决实验：CUDA graph / torch.compile 在本引擎的三问（先预测、后验证）。

H1 朴素 CUDA graph 捕获 decode forward：
   预测【失败】——MoE 的 offsets.tolist()（D2H 同步）在捕获期非法。
   若证实：这就是"先做 graph-safe MoE 才能上 graph"的实证（vLLM
   moe_align_block_size 存在的理由），而非猜测。

H2 30B@TP1 decode 的真实 GPU 空转率（同窗口测量）：
   修正 profile_v9 的方法学错误——旧版用【无 profiler 的步长】减
   【有 profiler 的 CUDA 时间】得出 -11ms 的荒谬值（负数间隙）。
   正确做法：profiler 窗口内同时量 wall 与 device time。
   参照：0.6B 上是 87% 空转（PERFORMANCE_NOTES），30B 待测——这决定
   CUDA graph 的收益上界。

H3 torch.compile(mode="reduce-overhead")：
   预测【guard 风暴】——prepare 每步重建 Context/张量 → 每步 guard 失败
   → 重编译或退化为 eager（v4 guard-tax 教训的全规模版）。
   测：首步编译耗时、后续逐步步长（风暴=步 2/3 又是分钟级）、
   recompile 计数、输出与 eager 的前缀一致率。

用法（每 phase 独立进程，避免 CUDA 状态互染）:
  PROBE_GPU=1 python tests/graph_probe.py h1
  PROBE_GPU=1 python tests/graph_probe.py h2
  PROBE_GPU=1 python tests/graph_probe.py h3
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __name__ != "__main__":
    raise SystemExit(0)                      # spawn 守卫

PATH = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
if not os.path.exists(f"{PATH}/config.json"):
    PATH = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"
PHASE = sys.argv[1] if len(sys.argv) > 1 else "h2"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", f"graph_probe_{PHASE}.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import torch
from transformers import AutoTokenizer
from samplingParams import SamplingParams
from engine.core import EngineCore

tok = AutoTokenizer.from_pretrained(PATH)
try:
    with open(f"{PATH}/chat_template.json") as f:
        tok.chat_template = json.load(f)["chat_template"]
except FileNotFoundError:
    pass
MSGS = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话",
        "用三句话介绍牛顿第二定律", "法国的首都是哪里", "水的化学式是什么"]
prompts = [tok.encode(tok.apply_chat_template(
    [{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True),
    add_special_tokens=False) for m in MSGS]

# h3 编译期 autotune 需要显存余量：bf16 权重 61G + ctx 6.4G，util 0.8 只剩
# 0.6G < 激活峰值 0.64G（实测 budget=-0.0GB 断言）——W4 量化路径（26.8G）
# 留足余量，结论对"编译×量化组合"同样有效。
USE_W4 = PHASE == "h3"
UTIL = 0.8 if USE_W4 else 0.85
res = {"phase": PHASE, "pred": {
    "h1": "朴素捕获失败于 MoE tolist（同步在捕获期非法）",
    "h2": "30B 空转率显著低于 0.6B 的 87%（kernel 更大）——具体值待测",
    "h3": "guard 风暴：每步 Context/张量重建 → guard 失败 → 重编译或退化 eager",
}.get(PHASE)}

core = EngineCore(PATH, tp_size=1,
                  gpus=[int(os.environ.get("PROBE_GPU", "1"))],
                  config_kwargs={"max_num_seqs": 8, "max_model_len": 2048,
                                 "max_num_batched_tokens": 16384,
                                 "gpu_memory_utilization": UTIL,
                                 "enforce_eager": True,
                                 "quantize_int4_w4": USE_W4})
sp = SamplingParams(temperature=0.0, max_tokens=48)
seqs_all = [core.add_request(p, sp) for p in prompts]


def steps(n, per_step=None):
    t0 = time.perf_counter()
    for _ in range(n):
        if core.is_finished():
            break
        ts = time.perf_counter()
        core.step()
        if per_step is not None:
            per_step.append(time.perf_counter() - ts)
    return time.perf_counter() - t0


try:
    if PHASE == "h1":
        # ---- 跑过 prefill，取 decode 稳态的固定 seqs ----
        steps(2)
        seqs = list(core.scheduler.running)
        runner = core.inproc_runner
        assert seqs and not seqs[0].is_prefill or True

        # 捕获 A：整个 run（含 sampler 的 tolist 与 MoE 的 tolist）
        g = torch.cuda.CUDAGraph()
        err_a = None
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):           # 协议要求的侧流热身
                for _ in range(2):
                    runner.run(seqs, False)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                runner.run(seqs, False)
        except Exception as e:
            err_a = f"{type(e).__name__}: {str(e)[:200]}"
        res["capture_full_run"] = "OK" if err_a is None else err_a

        # 捕获 B：仅 model forward（排除 sampler，仍含 MoE tolist）
        input_ids, positions = runner.prepare_decode(seqs)
        g2 = torch.cuda.CUDAGraph()
        err_b = None
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    runner.model(input_ids, positions)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g2):
                runner.model(input_ids, positions)
        except Exception as e:
            err_b = f"{type(e).__name__}: {str(e)[:200]}"
        res["capture_model_only"] = "OK" if err_b is None else err_b
        res["verdict"] = ("H1 证实：graph 捕获被阻断（见上）——graph-safe MoE "
                          "（去 tolist/去数据依赖循环）是 CUDA graph 的前置"
                          if err_a or err_b else "H1 证伪：可捕获！收益实验提上日程")

    elif PHASE == "h2":
        # ---- 无 profiler 基线（旧方法的分子，留作对照）----
        steps(8)                                  # 热身（JIT/选核）
        t_clean = steps(20)
        # ---- 同窗口测量：profiler 内同时量 wall 与 device ----
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            steps(20)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        ka = prof.key_averages()
        dev = sum(e.self_device_time_total for e in ka) / 1e6      # s
        res.update(unprofiled_wall_s=round(t_clean, 3),
                   profiled_wall_s=round(wall, 3),
                   device_busy_s=round(dev, 3),
                   gpu_idle_pct=round(100 * (1 - dev / wall), 1),
                   per_step_ms_profiled=round(wall / 20 * 1000, 1),
                   profiler_overhead_pct=round(100 * (wall / t_clean - 1), 1),
                   old_flawed_gap_ms=round((t_clean / 20 - dev / 20) * 1000, 1))
        res["verdict"] = (f"H2: profiled 窗口内 GPU 空转 {res['gpu_idle_pct']}% "
                          f"（0.6B 为 87%）——CUDA graph 收益上界 ≈ 消掉这部分间隙")

    elif PHASE == "h3":
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model = core.inproc_runner.model
        # reduce-overhead 实测：编译期物化 19.66GB 中间张量 OOM（MoE 循环被
        # 展开 + cudagraph 私有池）——降级 default 模式再测（无 cudagraph 池）
        mode = os.environ.get("COMPILE_MODE", "default")
        core.inproc_runner.model = torch.compile(model, mode=mode,
                                                  dynamic=False)
        per_step = []
        t0 = time.perf_counter()
        steps(10, per_step)                       # 首步含编译
        total = time.perf_counter() - t0
        outs = {}
        while not core.is_finished():
            for sid, toks in core.step():
                outs[sid] = toks
        # 与 eager 基线（quant_bf16_tp1 同负载）比对 token
        base_path = os.path.join(HERE, "results", "quant_bf16_tp1.json")
        match = None
        if os.path.exists(base_path):
            base = json.load(open(base_path))["outs"]
            sid2idx = {s.seq_id: i for i, s in enumerate(seqs_all)}
            m = 0
            for sid, toks in outs.items():
                b = base.get(str(sid2idx.get(sid, -1)), [])
                m += int(toks == b)
            match = f"{m}/{len(outs)}"
        try:
            from torch._dynamo.utils import counters
            stats = dict(counters.get("stats", {}))
        except Exception:
            stats = {}
        res.update(first10_total_s=round(total, 1),
                   per_step_s=[round(x, 2) for x in per_step],
                   eager_baseline_ms=544.0,          # W4 优化后基线（quant_w4_tp1）
                   output_match_vs_eager=match,
                   dynamo_stats=stats)
        res["verdict"] = ("H3: 首步编译 " + f"{per_step[0]:.0f}s；步 2-10 步长 "
                          f"{[round(x*1000) for x in per_step[1:]]}ms vs eager 146ms"
                          f"——{'guard 风暴（步长仍分钟级=重编译）' if any(x > 5 for x in per_step[1:]) else ('无增益（退化 eager）' if all(x*1000 > 100 for x in per_step[1:]) else '有增益')}")

except Exception as e:
    import traceback
    res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["traceback_tail"] = traceback.format_exc()[-800:]

finally:
    try:
        core.shutdown()
    except Exception:
        pass
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("RESULT" + json.dumps(res, ensure_ascii=False))
