"""v9.4e 混合批次 + chunk 无关性验证。

背景: scheduler 重写为 vLLM v1 的单循环追赶语义（无 prefill/decode 分支）——
任意请求可被 chunk（预算装不下就吃掉剩余），RUNNING 优先于 WAITING，
decode(1 token)与 prefill 尾段/首段混入同一批。

P0-17 修复: 旧版负载的"长 prompt"只有 ~100 token，远小于最小预算档 1024
——两个 phase 走完全相同的路径（单 chunk、无混批），"chunk 无关性"是平凡
通过，掩盖了 core→worker 切片协议的双重切片 bug。现在：
  1. prompt 用重复文本拉到 >3000 token，预算 1024 时必然切 3+ 段；
  2. 逐步记录 chunk 轮数，小预算 phase 断言 chunk 真的发生了（否则报
     "测试无效"而不是静默通过）；
  3. 混合批计数：长请求的 chunk 轮中与其他请求 decode 同批的轮数。

裁决: 小预算（1024, 长 prompt 被切成多段 + 与 decode 混批）vs 大预算
（16384, 无 chunk）的 greedy 输出逐 token 一致——chunk 边界不影响结果。
两 phase 用子进程隔离（单进程串行两个 EngineCore 会 OOM——allocator 残留）。
"""
import json, os, subprocess, sys

BASE = "/data/ifly_jyhe23/hot_model_top5/audio_understand/qwen-audio/vllm-learn/v9"
PATH = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
if not os.path.exists(f"{PATH}/config.json"):
    PATH = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"

WORKER = '''
import json, os, sys
sys.path.insert(0, "%s")
PATH = "%s"
from transformers import AutoTokenizer
from samplingParams import SamplingParams
tok = AutoTokenizer.from_pretrained(PATH)
try:
    with open(f"{PATH}/chat_template.json") as f:
        tok.chat_template = json.load(f)["chat_template"]
except FileNotFoundError:
    pass
base = "请详细介绍一下transformer架构的历史演进，从2017年的Attention is All You Need开始，包括BERT、GPT系列、以及近年来的MoE和长上下文的发展，尽量详细。"
long_text = base * 30                      # ~3000+ token: 预算 1024 下切 3+ 段
msgs = [long_text, "1+1等于几？只回答数字", "列举三种水果"]
ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                              add_generation_prompt=True) for m in msgs]
prompts = [tok.encode(p, add_special_tokens=False) for p in ps]
from engine.core import EngineCore
sp = SamplingParams(temperature=0.0, max_tokens=32)
core = EngineCore(PATH, tp_size=1, gpus=[0],
                  config_kwargs={"max_num_seqs": 8, "max_model_len": 8192,
                                 "max_num_batched_tokens": int(sys.argv[1]),
                                 "gpu_memory_utilization": 0.85, "enforce_eager": True})
sid2idx = {}
for i, p in enumerate(prompts):
    seq = core.add_request(p, sp)
    sid2idx[seq.seq_id] = i
outs = {}
steps = 0
chunk_rounds = []                          # 长请求的 chunk 轮（num_sched > 1）
mixed_rounds = 0                           # chunk 与他人 decode 同批的轮数
import time; t0 = time.time()
_orig_schedule = core.scheduler.schedule
def probe_schedule():
    seqs, is_p = _orig_schedule()
    long_sched = [s for s in seqs
                  if sid2idx.get(s.seq_id) == 0 and s.num_scheduled_tokens > 1]
    if long_sched:
        chunk_rounds.append(sum(s.num_scheduled_tokens for s in long_sched))
        if any(sid2idx.get(s.seq_id, -1) != 0 for s in seqs):
            mixed_rounds += 1
    return seqs, is_p
core.scheduler.schedule = probe_schedule
while not core.is_finished():
    for sid, toks in core.step():
        outs[sid2idx[sid]] = toks
    steps += 1
core.shutdown()
print("STEPS", steps, "RUN", round(time.time()-t0, 1))
print("CHUNKS", len(chunk_rounds), "MIXED", mixed_rounds, "LONGPROMPT", len(prompts[0]))
print("JSON" + json.dumps({str(k): v for k, v in outs.items()}))
''' % (BASE, PATH)

def run(budget):
    r = subprocess.run([sys.executable, "-c", WORKER, str(budget)],
                       capture_output=True, text=True, timeout=800)
    js = [l for l in r.stdout.splitlines() if l.startswith("JSON")][0][4:]
    st = [l for l in r.stdout.splitlines() if l.startswith("STEPS")][0]
    ch = [l for l in r.stdout.splitlines() if l.startswith("CHUNKS")][0]
    return json.loads(js), st, ch

small, st1, ch1 = run(1024)
big, st2, ch2 = run(16384)
print(st1, "| prompt 段预算 1024（chunk+混合批）")
print(st2, "| prompt 段预算 16384（无 chunk）")
print(ch1, "| 预算 1024 的 chunk 诊断")
print(ch2, "| 预算 16384 的 chunk 诊断（应 CHUNKS 0）")

# P0-17: 小预算 phase 必须真的发生了 chunked prefill，否则本测试无效
n_chunks = int(ch1.split()[1])
mixed = int(ch1.split()[3])
assert n_chunks >= 1, ("预算 1024 下长 prompt 没有被分段——负载没有触发 "
                       "chunked prefill，测试退化为平凡通过（P0-17 要防的形态）")
print(f"\n  混合批轮数: {mixed}（长请求的 chunk 轮中与他人 decode 同批的次数）")

print("\n===== chunk 无关性裁决 =====")
n = 0
for k in big:
    eq = small.get(k) == big[k]
    n += eq
    print(f"  req{k}: {'一致' if eq else '分歧'} ({len(small.get(k, []))}/{len(big[k])} tok)")
print(f"  {n}/{len(big)} 一致")
