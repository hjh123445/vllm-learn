import os
import torch
from engine.llm_engine import LLMEngine
from samplingParams import SamplingParams
from transformers import AutoTokenizer

path = os.environ.get("VLLM_LEARN_MODEL") or \
    "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
import json
tok = AutoTokenizer.from_pretrained(path)
# Omni 的 chat 模板在独立的 chat_template.json（不在 tokenizer_config 里）
with open(f"{path}/chat_template.json") as f:
    tok.chat_template = json.load(f)["chat_template"]
llm = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                gpu_memory_utilization=0.95, enforce_eager=True)
prompts = [
    "介绍一下你自己",
    "天空为什么是蓝色的？用一句话回答",
    "1+1等于几？只回答数字",
]
# 文本路径：直接用 chat 模板（不带音频/图像输入，thinker 退化为纯文本 LLM）
prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True)
           for p in prompts]
outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48))
for i, o in enumerate(outputs):
    print(f"===== 请求 {i} =====")
    print(o["text"][:200])
