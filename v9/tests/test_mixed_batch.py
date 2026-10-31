"""v9.4e 混合批次 + chunk 无关性验证。

背景: scheduler 重写为 vLLM v1 的单循环追赶语义（无 prefill/decode 分支）——
任意请求可被 chunk（预算装不下就吃掉剩余），RUNNING 优先于 WAITING，
decode(1 token)与 prefill 尾段/首段混入同一批。

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
msgs = ["请详细介绍一下transformer架构的历史演进，从2017年的Attention is All You Need开始，包括BERT、GPT系列、以及近年来的MoE和长上下文的发展，尽量详细。",
        "1+1等于几？", "列举三种水果"]
ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                              add_generation_prompt=True) for m in msgs]
prompts = [tok.encode(p, add_special_tokens=False) for p in ps]
from engine.core import EngineCore
sp = SamplingParams(temperature=0.0, max_tokens=32)
core = EngineCore(PATH, tp_size=1, gpus=[0],
                  config_kwargs={"max_num_seqs": 8, "max_model_len": 2048,
                                 "max_num_batched_tokens": int(sys.argv[1]),
                                 "gpu_memory_utilization": 0.85, "enforce_eager": True})
sid2idx = {}
for i, p in enumerate(prompts):
    seq = core.add_request(p, sp)
    sid2idx[seq.seq_id] = i
outs = {}
steps = 0
import time; t0 = time.time()
while not core.is_finished():
    for sid, toks in core.step():
        outs[sid2idx[sid]] = toks
    steps += 1
core.shutdown()
print("STEPS", steps, "RUN", round(time.time()-t0, 1))
print("JSON" + json.dumps({str(k): v for k, v in outs.items()}))
''' % (BASE, PATH)

def run(budget):
    r = subprocess.run([sys.executable, "-c", WORKER, str(budget)],
                       capture_output=True, text=True, timeout=800)
    js = [l for l in r.stdout.splitlines() if l.startswith("JSON")][0][4:]
    st = [l for l in r.stdout.splitlines() if l.startswith("STEPS")][0]
    return json.loads(js), st

small, st1 = run(1024)
big, st2 = run(16384)
print(st1, "| prompt 段预算 1024（chunk+混合批）")
print(st2, "| prompt 段预算 16384（无 chunk）")
print("\n===== chunk 无关性裁决 =====")
n = 0
for k in big:
    eq = small.get(k) == big[k]
    n += eq
    print(f"  req{k}: {'一致' if eq else '分歧'} ({len(small.get(k, []))}/{len(big[k])} tok)")
print(f"  {n}/{len(big)} 一致")
