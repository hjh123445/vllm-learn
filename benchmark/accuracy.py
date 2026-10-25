# benchmark/accuracy.py
"""正确性测试套件。v0/v1 模块名相同，每个测试在独立子进程里跑 worker。

用法:
  python benchmark/accuracy.py              # 全部测试
  python benchmark/accuracy.py --test 2    # 只跑测试 2

worker 模式（被驱动进程调用）:
  python benchmark/accuracy.py --run T1 --engine v0 ...
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 模型路径：环境变量 VLLM_LEARN_MODEL 优先；默认相对仓库根的上一级
MODEL_PATH = os.environ.get("VLLM_LEARN_MODEL") or os.path.join(ROOT, "..", "Qwen3-0.6B")
QUESTIONS = [
    "天空为什么是蓝色的？用一句话回答",
    "你好",
    "介绍你自己",
    "1+1等于几？只回答数字",
]

# ============================================================ worker

def worker(args):
    """在独立进程里跑单个测试，结果以 JSON 打到 stdout。"""
    sys.path.insert(0, os.path.join(ROOT, args.engine))
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
               for q in QUESTIONS]

    result = {"test": args.run, "engine": args.engine}
    # v2 会按 gpu_memory_utilization 划池子；测试机上设小一点
    ekw = {"gpu_memory_utilization": 0.25} if args.engine == "v2" else {}

    if args.run == "T1":       # greedy 确定性
        from engine.engine import Engine
        from samplingParams import SamplingParams
        llm = Engine(MODEL_PATH, batch_size=8, max_seq_len=4096, dtype=torch.bfloat16, **ekw)
        sp = SamplingParams(temperature=0.0, max_tokens=32)
        a = llm.generate([prompts[0]], sp)[0]["token_ids"]
        b = llm.generate([prompts[0]], sp)[0]["token_ids"]
        result["pass"] = a == b
        result["detail"] = f"{len(a)} vs {len(b)} tokens"

    elif args.run == "T2":     # batch 一致性：batch 里跑 == 单独跑
        # 用 fp32：bf16 下 batch 形状不同会让 kernel 归约顺序不同，长输出会数值性分歧
        # （不是逻辑 bug）。fp32 把数值噪声压掉，只留逻辑错误。
        from engine.engine import Engine
        from samplingParams import SamplingParams
        llm = Engine(MODEL_PATH, batch_size=8, max_seq_len=4096, dtype=torch.float32, **ekw)
        sp = SamplingParams(temperature=0.0, max_tokens=32)
        batched = llm.generate(prompts, sp)
        result["pass"] = True
        detail = []
        for i, p in enumerate(prompts):
            alone = llm.generate([p], sp)[0]["token_ids"]
            same = alone == batched[i]["token_ids"]
            result["pass"] &= same
            detail.append(f"req{i}:{'OK' if same else 'DIFF'}")
        result["detail"] = "(fp32) " + " ".join(detail)

    elif args.run == "T3":     # 单请求生成（给 T3 驱动做 v0/v1 对比）
        from engine.engine import Engine
        from samplingParams import SamplingParams
        llm = Engine(MODEL_PATH, batch_size=8, max_seq_len=4096, dtype=torch.bfloat16, **ekw)
        sp = SamplingParams(temperature=0.0, max_tokens=48)
        result["outputs"] = [llm.generate([p], sp)[0]["token_ids"] for p in prompts]
        result["pass"] = all(len(o) == 48 for o in result["outputs"])

    elif args.run == "T4":     # HF transformers 对齐（fp32 逐 token）
        # bf16 下双方 kernel 数值噪声会把 argmax 翻掉（前缀一致 23~32/32），
        # fp32 下实现正确应逐 token 一致（修复 eps + in-place bug 后实测 32/32）
        from engine.engine import Engine
        from samplingParams import SamplingParams
        from transformers import AutoModelForCausalLM
        llm = Engine(MODEL_PATH, batch_size=8, max_seq_len=4096, dtype=torch.float32, **ekw)
        sp = SamplingParams(temperature=0.0, max_tokens=32)
        ours = [llm.generate([p], sp)[0]["token_ids"] for p in prompts]

        hf = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float32, device_map="cuda")
        hf.eval()
        detail, all_prefix = [], []
        with torch.no_grad():
            for i, p in enumerate(prompts):
                ids = tok(p, return_tensors="pt").input_ids.cuda()
                ref = hf.generate(ids, attention_mask=torch.ones_like(ids),
                                  max_new_tokens=32, do_sample=False)[0][ids.shape[1]:]
                ref = ref.tolist()
                same = 0
                for a, b in zip(ref, ours[i]):
                    if a != b:
                        break
                    same += 1
                all_prefix.append(same / max(len(ref), 1))
                detail.append(f"req{i}:{same}/{len(ref)}")
        result["pass"] = all(x >= 0.9 for x in all_prefix)
        result["detail"] = "(fp32) " + " ".join(detail) + f" (avg {sum(all_prefix)/len(all_prefix):.0%})"

    print(json.dumps(result, ensure_ascii=False))


# ============================================================ driver

def spawn(run, engine, **kw):
    cmd = [sys.executable, __file__, "--run", run, "--engine", engine]
    for k, v in kw.items():
        cmd += [f"--{k.replace('_','-')}", str(v)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return {"pass": False, "detail": f"worker 崩溃: {out.stderr.strip().splitlines()[-1]}"}
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["T1", "T2", "T3", "T4"])
    ap.add_argument("--engine", choices=["v0", "v1", "v2"])
    ap.add_argument("--test", type=int)
    args = ap.parse_args()
    if args.run:
        return worker(args)

    results = []
    ENGINES = ("v0", "v1", "v2")
    if not args.test or args.test == 1:
        for e in ENGINES:
            results.append((f"T1 greedy 确定性 [{e}]", spawn("T1", e)))
    if not args.test or args.test == 2:
        for e in ENGINES:
            results.append((f"T2 batch 一致性 [{e}]", spawn("T2", e)))
    if not args.test or args.test == 3:
        outs = [spawn("T3", e) for e in ENGINES]
        ref = outs[0].get("outputs", [])
        same = all(o.get("outputs") == ref for o in outs[1:])
        ok = all(o.get("pass") for o in outs)
        results.append(("T3 v0/v1/v2 等价（单请求路径）",
                        {"pass": bool(same and ok),
                         "detail": f"4 条请求逐 token 对比: {'全部一致' if same else '存在分歧'}"}))
    if not args.test or args.test == 4:
        results.append(("T4 HF transformers 对齐 [v2]", spawn("T4", "v2")))

    n_pass = sum(1 for _, r in results if r["pass"])
    print(f"\n===== accuracy: {n_pass}/{len(results)} PASS =====")
    for name, r in results:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {name}  {r.get('detail', '')}")
    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
