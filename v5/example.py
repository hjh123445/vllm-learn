import os
import torch
from engine.llm_engine import LLMEngine as LLM
from samplingParams import SamplingParams

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Qwen3-0.6B")
llm = LLM(path, enforce_eager=True, max_num_seqs=8, max_model_len=4096, gpu_memory_utilization=0.3)
prompts = [
    "introduce yourself",
    "为什么天空是蓝色的",
    "人活着的意义是什么",
]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path)
prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True) for p in prompts]
outputs = llm.generate(prompts, SamplingParams(temperature=1e-6, max_tokens=128))
for i, o in enumerate(outputs):
    print(f"===== 请求 {i} =====")
    print(o["text"])
