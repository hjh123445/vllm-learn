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
prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt} for prompt in prompts],
                                        tokenize=False, add_generation_prompt=True)]
outputs = llm.generate(prompts, SamplingParams(temperature=0.6, max_tokens=256))
print(outputs[0]["text"])