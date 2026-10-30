"""v9.1 e2e：W4 量化引擎 vs bf16 基线——显存实测 + L3 端到端裁决。

裁决标准（DESIGN.md §II.3.4，量化是有损的，不用逐位一致）：
  - 权重显存 -75% 账面实测
  - greedy token 前缀一致长度 + 分歧率（vs bf16 引擎同 prompt）
  - 数值参考：单层 hidden MSE（L2 级）
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from engine.llm_engine import LLMEngine
from samplingParams import SamplingParams
from transformers import AutoTokenizer

path = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
tok = AutoTokenizer.from_pretrained(path)
with open(f"{path}/chat_template.json") as f:
    tok.chat_template = json.load(f)["chat_template"]


def weight_bytes(model) -> int:
    """模型参数总字节（Parameter + 注册 buffer）。"""
    n = 0
    for p in model.parameters():
        n += p.numel() * p.element_size() if p is not None else 0
    for p in model.buffers():
        n += p.numel() * p.element_size()
    return n


def run(tag, quantize=False):
    eng = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                    gpu_memory_utilization=0.95, enforce_eager=True)
    before = weight_bytes(eng.model_runner.model)
    if quantize:
        eng.model_runner.model.quantize_int4(group=128)
        torch.cuda.empty_cache()
    after = weight_bytes(eng.model_runner.model)
    print(f"\n[{tag}] 权重存储: {before/1e9:.2f} GB → {after/1e9:.2f} GB "
          f"({before/after:.2f}x 压缩)", flush=True)

    msgs = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话",
            "写一个 Python 快速排序", "翻译成英文：今天天气真好"]
    ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                                  add_generation_prompt=True) for m in msgs]
    t0 = time.time()
    outs = eng.generate(ps, SamplingParams(temperature=0.0, max_tokens=48))
    wall = time.time() - t0
    n_tok = sum(len(o["token_ids"]) for o in outs)
    print(f"[{tag}] wall={wall:.1f}s thr={n_tok/wall:.1f} tok/s", flush=True)
    del eng
    torch.cuda.empty_cache()
    return outs, wall


if __name__ == "__main__":
    base, t_base = run("bf16")
    quant, t_quant = run("W4-int4-g128", quantize=True)

    print("\n===== v9.1 L3 端到端裁决（量化 vs bf16）=====")
    total, agree, first_div = 0, 0, []
    for i, (b, q) in enumerate(zip(base, quant)):
        bt, qt = b["token_ids"], q["token_ids"]
        n = min(len(bt), len(qt))
        eq = sum(1 for a, c in zip(bt, qt) if a == c)
        total += n
        agree += eq
        fd = next((j for j in range(n) if bt[j] != qt[j]), n)
        first_div.append(fd)
        print(f"  req{i}: 一致 {eq}/{n} (首个分歧 @tok{fd})")
        print(f"    bf16: {b['text'][:40]!r}")
        print(f"    W4  : {q['text'][:40]!r}")
    rate = agree / total
    print(f"\n  总分歧率: {1-rate:.1%}（合格线 <5%，DESIGN §II.3.4）")
    print(f"  最短前缀一致: {min(first_div)} token")
    print(f"  速度: bf16 {t_base:.1f}s vs W4(B1) {t_quant:.1f}s "
          f"（B1 慢是预期内，v9.2 kernel 化后兑现）")
    print("\n===== v9.1 完成 =====")
