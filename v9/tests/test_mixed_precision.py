"""Qwen3-Omni-30B 混合精度：专家 W4，attention/embedding/lm_head 保持 BF16。"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer
from engine.llm_engine import LLMEngine
from samplingParams import SamplingParams

PATH = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
tok = AutoTokenizer.from_pretrained(PATH)
with open(f"{PATH}/chat_template.json", encoding="utf-8") as f:
    tok.chat_template = json.load(f)["chat_template"]
msgs = ["介绍一下你自己", "天空为什么是蓝色的？一句话", "翻译成英文：今天天气真好"]
prompts = [tok.apply_chat_template([{"role":"user", "content":m}],
                                    tokenize=False, add_generation_prompt=True)
           for m in msgs]

def bytes_of(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) + \
           sum(b.numel() * b.element_size() for b in model.buffers())

def run(quant=False):
    eng = LLMEngine(PATH, max_num_seqs=8, max_model_len=2048,
                    max_num_batched_tokens=512, gpu_memory_utilization=0.90,
                    enforce_eager=True)
    before = bytes_of(eng.model_runner.model)
    if quant:
        eng.model_runner.model.quantize_int4(group=128, skip_dense=True)
        torch.cuda.empty_cache()
    after = bytes_of(eng.model_runner.model)
    t0 = time.time()
    out = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=24))
    wall = time.time() - t0
    del eng
    torch.cuda.empty_cache()
    return out, before, after, wall

if __name__ == "__main__":
    base, b0, b1, tb = run(False)
    mix, q0, q1, tq = run(True)
    eq = sum(a == b for x, y in zip(base, mix)
             for a, b in zip(x["token_ids"], y["token_ids"]))
    total = sum(min(len(x["token_ids"]), len(y["token_ids"]))
                for x, y in zip(base, mix))
    print(json.dumps({"bf16_gb": b1/1e9, "mixed_gb": q1/1e9,
                      "compression": b1/q1, "bf16_s": tb, "mixed_s": tq,
                      "bf16_tok_s": sum(len(x["token_ids"]) for x in base)/tb,
                      "mixed_tok_s": sum(len(x["token_ids"]) for x in mix)/tq,
                      "agree": eq/max(total,1)}, ensure_ascii=False))
