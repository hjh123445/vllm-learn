"""v9 单卡 decode 的 profile：定位每步时间构成（面试热点归因数据）。

输出:
  - 每步耗时分解（CPU total / CUDA total / 间隙）
  - top-15 CUDA 算子（自用时长）
  - top-15 CPU 算子（Self CPU——launch 开销的指纹）
  - kernel 数量/步、launch 均价估算
结果存 results/profile_v9_tp1.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
if not os.path.exists(f"{PATH}/config.json"):
    PATH = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"

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

core = EngineCore(PATH, tp_size=1,
                  gpus=[int(os.environ.get("PROFILE_GPU", "0"))],
                  config_kwargs={"max_num_seqs": 8, "max_model_len": 2048,
                                 "max_num_batched_tokens": 16384,
                                 "gpu_memory_utilization": 0.85, "enforce_eager": True})
sp = SamplingParams(temperature=0.0, max_tokens=48)

import torch
from torch.profiler import profile, ProfilerActivity

# ---- 阶段 1: 热身 12 步（JIT/选核，不计入）----
seqs = [core.add_request(p, sp) for p in prompts]
for _ in range(12):
    if core.is_finished():
        break
    core.step()

# ---- 阶段 2: 计时（profiler 关闭，免 observer 开销）----
step_times = []
while not core.is_finished():
    t0 = time.perf_counter()
    core.step()
    step_times.append(time.perf_counter() - t0)
avg = sum(step_times) / max(len(step_times), 1)
print(f"timed steps={len(step_times)} avg={avg*1e3:.1f}ms")

# ---- 阶段 3: profiler 段（新一批请求，同 EngineCore——旧版第二次构造
# EngineCore 在共享 GPU 上会叠加两次 0.85util 权重直接 OOM）----
seqs = [core.add_request(p, sp) for p in prompts]
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    k = 0
    while not core.is_finished() and k < 20:
        core.step()
        k += 1
core.shutdown()

ka = prof.key_averages()
rows_cuda = sorted(ka, key=lambda e: e.self_device_time_total, reverse=True)
rows_cpu = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)
total_cuda = sum(e.self_device_time_total for e in ka) / 1e3 / max(k, 1)   # ms/步
total_cpu = sum(e.self_cpu_time_total for e in ka) / 1e3 / max(k, 1)
n_events = sum(e.count for e in ka) / max(k, 1)

def fmt(e):
    return {"op": e.key[:60], "count_per_step": e.count / max(k, 1),
            "self_cuda_us_per_step": e.self_device_time_total / max(k, 1),
            "self_cpu_us_per_step": e.self_cpu_time_total / max(k, 1)}

out = {"avg_step_ms": avg * 1e3, "profiled_steps": k,
       "total_cuda_ms_per_step": total_cuda,
       "total_cpu_ms_per_step": total_cpu,
       "gpu_gap_ms_per_step": avg * 1e3 - total_cuda,
       "events_per_step": n_events,
       "top_cuda": [fmt(e) for e in rows_cuda[:15]],
       "top_cpu": [fmt(e) for e in rows_cpu[:15]]}
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                       "profile_v9_tp1.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"cuda/step={total_cuda:.1f}ms cpu/step={total_cpu:.1f}ms "
      f"gpu_gap={out['gpu_gap_ms_per_step']:.1f}ms events/step={n_events:.0f}")
print("top-5 CUDA:")
for e in rows_cuda[:5]:
    print(f"  {e.key[:55]:<55} {e.self_device_time_total/k:>8.0f} us/step  x{e.count/k:.0f}")
print("top-5 CPU:")
for e in rows_cpu[:5]:
    print(f"  {e.key[:55]:<55} {e.self_cpu_time_total/k:>8.0f} us/step  x{e.count/k:.0f}")
