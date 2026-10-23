from engine.llm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import torch

path = os.path.expanduser("/data/ifly_jyhe23/hot_model_top5/audio_understand/qwen-audio/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, batch_size=8, max_seq_len=4096, dtype=torch.bfloat16)
prompts = [
    "introduce yourself",
    "为什么天空是蓝色的",
    "人活着的意义是什么"
]
# 每条 prompt 独立套一次 chat template，各自成一段完整对话
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
           for p in prompts]
outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=2048))
for i, o in enumerate(outputs):
    print(f"===== 请求 {i} =====")
    print(o["text"])
    print()
