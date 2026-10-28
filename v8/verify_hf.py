# v6/verify_hf.py
"""v6.0 正确性验证：thinker 文本路径与 HF Qwen3OmniMoe 逐 token 对比（fp32）。

注意：fp32 下模型要 132GB 单卡放不下 → 用 bf16 对比（宽松容差）+
短 prompt 逐 token 前缀一致率。这是 MoE 路径的首个对齐验证。
"""
import os
import sys

import torch

path = os.environ.get("VLLM_LEARN_MODEL") or \
    "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"

# ---- 我们的引擎 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.llm_engine import LLMEngine
from samplingParams import SamplingParams
from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained(path)
with open(f"{path}/chat_template.json") as f:
    tok.chat_template = json.load(f)["chat_template"]

QUESTIONS = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话"]

prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)
           for q in QUESTIONS]

llm = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                gpu_memory_utilization=0.95, enforce_eager=True)
ours = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))

# ---- HF 参考（bf16, 同 GPU）----
del llm
torch.cuda.empty_cache()

from transformers import Qwen3OmniMoeForConditionalGeneration

hf = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    path, torch_dtype=torch.bfloat16, device_map="cuda:0")
hf.eval()

n_pass = 0
with torch.no_grad():
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        out = hf.generate(ids, attention_mask=torch.ones_like(ids),
                          max_new_tokens=32, do_sample=False)
        # Omni 的 generate 返回 tuple: (thinker_token_ids, audio_codes)
        ref = out[0][0][ids.shape[1]:].tolist()
        our = ours[i]["token_ids"]
        same = 0
        for a, b in zip(ref, our):
            if a != b:
                break
            same += 1
        ok = same >= len(ref) * 0.9
        n_pass += ok
        print(f"req{i}: 前缀一致 {same}/{len(ref)} {'PASS' if ok else 'FAIL'}")
print(f"===== v6.0 HF 对齐: {n_pass}/3 =====")
