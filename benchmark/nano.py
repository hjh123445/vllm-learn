# benchmark/nano.py
"""在 nano-vllm 上跑同一套负载和同一套方法学，和我们的 v0/v1/v2 对比。

用法: python benchmark/nano.py --num-requests 12 --max-num-seqs 8
要求: nano-vllm 在 ../nano-vllm（仓库根的上一级），flash-attn 已编译安装。
负载与 harness.py 完全一致（固定问题集 + 32/96/160 混合 max_tokens），
计时口径同 speed.py: TTFT / TPOT / E2E / 吞吐，用 engine.step() 逐步打时间戳。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NANO = os.path.join(os.path.dirname(ROOT), "nano-vllm")
sys.path.insert(0, NANO)
sys.path.insert(0, HERE)

from harness import MODEL_PATH, QUESTIONS, MAX_TOKEN_TIERS  # noqa: E402  同一套负载


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-requests", type=int, default=12)
    ap.add_argument("--max-num-seqs", type=int, default=8,
                    help="并发上限（对齐我们的 batch_size 口径）")
    ap.add_argument("--gpu-mem-util", type=float, default=0.3,
                    help="对齐我们 v2 的池子预算口径")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="关掉 CUDA graph（对比 launch 开销）")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(MODEL_PATH, enforce_eager=args.enforce_eager,
              max_num_seqs=args.max_num_seqs, max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem_util)

    # 与 harness.build_workload_impl 完全一致的负载
    prompts, sps = [], []
    for i in range(args.num_requests):
        q = QUESTIONS[i % len(QUESTIONS)]
        text = tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
        prompts.append(tok.encode(text, add_special_tokens=False))
        sps.append(SamplingParams(temperature=1e-6,   # nano 禁止 greedy（assert temperature>1e-10），1e-6 的 softmax 是 one-hot，等价 greedy
                                  max_tokens=MAX_TOKEN_TIERS[i % len(MAX_TOKEN_TIERS)]))

    engine = llm    # LLM 直接继承 LLMEngine，engine 就是它自己
    t0 = time.perf_counter()
    first_t, finish_t = {}, {}
    my_ids = []
    for p, sp in zip(prompts, sps):
        engine.add_request(p, sp)
        my_ids.append(engine.scheduler.waiting[-1].seq_id)   # 入队后立刻抓自己的 seq_id
    id2i = {sid: i for i, sid in enumerate(my_ids)}
    # LLM 构造时已做 warmup profile；逐步推进。step() 返回本步完成的序列（postprocess
    # 已把它们移出队列），first token 靠扫描在途序列记录。
    while not engine.is_finished():
        outputs, _ = engine.step()
        now = time.perf_counter()
        for sid, _comp in outputs:
            if sid in id2i and sid not in finish_t:
                finish_t[sid] = now - t0
        for seq in list(engine.scheduler.waiting) + list(engine.scheduler.running):
            if seq.seq_id in id2i and seq.seq_id not in first_t and len(seq.completion_token_ids) > 0:
                first_t[seq.seq_id] = now - t0
    total = time.perf_counter() - t0

    ids = my_ids
    ttft = [first_t.get(i, 0) * 1e3 for i in ids]
    e2e = [finish_t.get(i, first_t.get(i, 0)) * 1e3 for i in ids]
    tpot = [(finish_t[i] - first_t[i]) / (sps[id2i[i]].max_tokens - 1) * 1e3 for i in ids]
    total_out = sum(sps[id2i[i]].max_tokens for i in ids)

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(int(len(xs) * p), len(xs) - 1)]

    out = {
        "engine": "nano-vllm",
        "num_requests": args.num_requests,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "total_time": round(total, 2),
        "ttft_avg_ms": round(sum(ttft) / len(ttft), 1),
        "ttft_p50_ms": round(pct(ttft, 0.5), 1),
        "ttft_max_ms": round(max(ttft), 1),
        "tpot_avg_ms": round(sum(tpot) / len(tpot), 1),
        "e2e_p50_ms": round(pct(e2e, 0.5), 1),
        "e2e_max_ms": round(max(e2e), 1),
        "total_out_tokens": total_out,
        "tok_per_s": round(total_out / total, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    out_path = os.path.join(HERE, "results", "speed_nano.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
