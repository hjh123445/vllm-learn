"""v7 投机解码验证：与普通 greedy 逐 token 对比 + 接受率统计。
负载特意选重复性高的文本（n-gram draft 的甜区）+ 普通文本对照。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from engine.llm_engine import LLMEngine
from samplingParams import SamplingParams
from layers.mtp import NGramDrafter, SpeculativeEngine
import json
from transformers import AutoTokenizer

path = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
tok = AutoTokenizer.from_pretrained(path)
with open(f"{path}/chat_template.json") as f:
    tok.chat_template = json.load(f)["chat_template"]

eng = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                gpu_memory_utilization=0.95, enforce_eager=True)

# 重复性高的 prompt（n-gram draft 甜区）和普通 prompt 对照
PROMPTS = [
    "数数：1 2 3 4 5 6 7 8 9 10 11 12",           # 高重复
    "重复这句话三遍：今天天气真好。",               # 高重复
    "介绍一下你自己",                              # 普通文本
]
NEW = 48

for i, msg in enumerate(PROMPTS):
    p = tok.apply_chat_template([{"role": "user", "content": msg}],
                                tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(p, add_special_tokens=False)

    # 基线: 普通 greedy
    t0 = time.time()
    base = eng.generate([p], SamplingParams(temperature=0.0, max_tokens=NEW))[0]["token_ids"]
    t_base = time.time() - t0

    # 投机版
    spec = SpeculativeEngine(eng, NGramDrafter(n=2, k=4))
    t0 = time.time()
    spec_out = spec.generate(prompt_ids, max_tokens=NEW)
    t_spec = time.time() - t0

    same = base[:len(spec_out)] == spec_out[:len(base)]
    print(f"--- req{i} ({msg[:12]}...) 基线 {t_base:.1f}s vs 投机 {t_spec:.1f}s "
          f"输出一致: {same}")
    print(f"    接受率: {spec.accept_rate:.0%}  (rounds={spec.n_rounds}, "
          f"accepted={spec.n_accepted}/{spec.n_rounds*4})")
    print(f"    文本: {tok.decode(spec_out[:24], skip_special_tokens=True)!r}")
