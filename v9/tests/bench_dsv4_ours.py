"""我们引擎(v6.3)跑 DSV4-Flash 8 卡分层分放 + bench + profile。

配置: 43 层 × 8 卡(6,6,6,6,5,5,5,4 类似 dsv41 配方), bf16 反量化,
      跨卡层间搬运激活, eager, greedy。
Bench: 3 prompts × 32 tok, 记录 load/gen 时间与 tok/s。
Profile: torch profiler 抓 8 步 decode 的 chrome trace。
"""
import os, sys, time, json
sys.path.insert(0, "/data/ifly_jyhe23/hot_model_top5/audio_understand/qwen-audio/vllm-learn/v6")
import torch

MODEL = "/data/xmhuang6/deepseek/DeepSeek-V4-Flash"
from models.deepseek_v4 import DSV4ForCausalLM
from utils.dsv4_loader import load_dsv4
from transformers import AutoTokenizer

n_layers = 43
layer_map = {}
rest = list(range(4, n_layers))
per = (len(rest) + 6) // 7
for i in range(n_layers):
    layer_map[i] = "cuda:0" if i < 4 else f"cuda:{1 + (i - 4) // per}"
layer_map[-1] = "cuda:7"

t0 = time.time()
model = DSV4ForCausalLM(MODEL)
load_dsv4(model, MODEL, device="cuda:0", dtype=torch.bfloat16, layer_device_map=layer_map)
t_load = time.time() - t0
print(f"[ours-load] {t_load:.0f}s", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
prompts = ["The capital of France is",
           "请介绍transformer架构的核心思想",
           "Write a Python quicksort function"]

def gen(ids, n_new, profile_path=None):
    cur = list(ids); pos = list(range(len(cur)))
    out = []
    import torch.nn.functional as F
    prof_ctx = None
    if profile_path:
        from torch.profiler import profile, ProfilerActivity
        prof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof_ctx.__enter__()
    t0 = time.time()
    for step in range(n_new):
        x = torch.tensor(cur, device="cuda:0")
        p = torch.arange(len(cur), device="cuda:0")
        with torch.no_grad():
            h = model(x, p)
            logits = model.compute_logits(h[-1:])
        nxt = int(logits.argmax(-1).item())
        out.append(nxt); cur.append(nxt)
    dt = time.time() - t0
    if prof_ctx is not None:
        prof_ctx.__exit__(None, None, None)
        prof_ctx.export_chrome_trace(profile_path)
    return out, dt

results = {}
for i, ptext in enumerate(prompts):
    ids = tok.encode(ptext, add_special_tokens=False)
    out, dt = gen(ids, 32)
    results[ptext[:20]] = {"tokens": len(out), "time_s": round(dt, 1),
                           "tok_s": round(len(out)/dt, 2)}
    print(f"[ours] req{i}: {len(out)} tok in {dt:.1f}s = {len(out)/dt:.2f} tok/s "
          f"| {tok.decode(out[:12], skip_special_tokens=True)!r}", flush=True)

# profile 8 步
ids = tok.encode(prompts[0], add_special_tokens=False)
gen(ids[:8], 8, profile_path="/tmp/ours_dsv4_trace.json")
print("[profile] saved /tmp/ours_dsv4_trace.json", flush=True)
json.dump(results, open("/tmp/ours_dsv4_bench.json", "w"), ensure_ascii=False, indent=1)
print("[done]")
