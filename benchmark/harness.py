# benchmark/harness.py
"""计时 harness：不走 Engine.generate 黑盒，直接用 Scheduler + ModelRunner 组件复刻
v0/v1 的生成循环，在关键点打时间戳。同时也是唯一的 worker 进程（各脚本通过子进程调用，
避免 v0/v1 模块名冲突不能同进程导入）。

CLI:
  python harness.py --engine v0 --num-requests 12 --batch-size 8 [--profile]
输出 JSON（stdout 或 --out 文件）：
  {requests: [...], steps: [...], stage_times: {...}, outputs: [...], meta: {...}}
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 模型路径：环境变量 VLLM_LEARN_MODEL 优先；默认相对仓库根的上一级（跟仓库走，不锁死机器路径）
MODEL_PATH = os.environ.get("VLLM_LEARN_MODEL") or os.path.join(ROOT, "..", "Qwen3-0.6B")

# 固定负载问题集（可复现），max_tokens 三挡轮转模拟混合长度流量
QUESTIONS = [
    "1+1等于几？只回答数字",
    "用一句话解释什么是瑞利散射",
    "介绍你自己",
    "天空为什么是蓝色的？用一句话回答",
    "水的化学式是什么",
    "用三句话介绍牛顿第二定律",
    "你好",
    "光速是多少",
    "用一句话说明为什么冬天昼短夜长",
    "法国的首都是哪里",
    "简述什么是熵",
    "用两句话解释彩虹的成因",
]
MAX_TOKEN_TIERS = [32, 96, 160]


def build_workload_impl(num_requests, tokenizer, SamplingParams):
    prompts, sps = [], []
    for i in range(num_requests):
        q = QUESTIONS[i % len(QUESTIONS)]
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        prompts.append(tokenizer.encode(text, add_special_tokens=False))
        sps.append(SamplingParams(temperature=0.0, max_tokens=MAX_TOKEN_TIERS[i % len(MAX_TOKEN_TIERS)]))
    return prompts, sps


# ---------------------------------------------------------------- v0

def run_v0(runner_cls, scheduler_cls, prompt_ids, sps, batch_size, max_seq_len, eos_ids, pad_id):
    import torch
    runner = runner_cls(MODEL_PATH, batch_size=batch_size, max_seq_len=max_seq_len)
    sch = scheduler_cls(eos_id=next(iter(eos_ids)), pad_id=pad_id,
                        max_batch_size=batch_size, max_len=max_seq_len)
    stage = {"prefill": 0.0, "decode": 0.0, "sample": 0.0, "mask": 0.0}
    t0 = time.perf_counter()
    reqs = []          # 每个请求的记录
    steps = []         # 每个 decode 迭代 (t, 活跃数, batch 内未完成数)

    for start in range(0, len(prompt_ids), batch_size):
        chunk_ids = prompt_ids[start:start + batch_size]
        chunk_sps = sps[start:start + batch_size]
        batch_start = time.perf_counter()
        sch.add(chunk_ids, chunk_sps)
        batch = sch.schedule()
        runner.reset_cache()

        recs = [{"first_token_t": None, "finish_t": None,
                 "batch_start": batch_start - t0, "batch_end": None,
                 "n_tokens": 0} for _ in chunk_ids]

        sp = batch.params[0]
        while not batch.is_finished():
            ids, pos = batch.get()
            ts = time.perf_counter()
            mask = (runner.prefill_mask(batch) if batch.is_prefill
                    else runner.decode_mask(batch))
            stage["mask"] += time.perf_counter() - ts
            key = "prefill" if batch.is_prefill else "decode"
            ts = time.perf_counter()
            logits = runner.forward(ids, pos, mask)
            stage[key] += time.perf_counter() - ts
            ts = time.perf_counter()
            next_tok = runner.sample(logits, sp.temperature, 1.0)
            stage["sample"] += time.perf_counter() - ts
            was_prefill = batch.is_prefill
            before = list(batch.finished)
            batch.append(next_tok)
            now = time.perf_counter()
            if was_prefill:
                for r in recs:
                    r["first_token_t"] = now - t0
            else:
                for j, (was, now_f) in enumerate(zip(before, batch.finished)):
                    if not was and now_f:
                        recs[j]["finish_t"] = now - t0
                steps.append({"t": now - t0, "n_running": len(chunk_ids),
                              "n_active": sum(not f for f in batch.finished)})

        batch_end = time.perf_counter()
        collected = batch.collect()
        for j, r in enumerate(recs):
            r["batch_end"] = batch_end - t0
            r["n_tokens"] = len(collected[j]["token_ids"])
            r["output"] = collected[j]["token_ids"]
        reqs.extend(recs)

    return {"requests": reqs, "steps": steps, "stage_times": stage,
            "total_time": time.perf_counter() - t0}


# ---------------------------------------------------------------- v1

def run_v1(runner_cls, scheduler_cls, prompt_ids, sps, batch_size, max_seq_len, eos_ids):
    runner = runner_cls(MODEL_PATH, batch_size=batch_size, max_seq_len=max_seq_len)
    sch = scheduler_cls(eos_ids=eos_ids, max_seq_len=max_seq_len, max_batch_size=batch_size)
    stage = {"mixed": 0.0}   # v1 混合批次：prefill+decode 一个 forward
    t0 = time.perf_counter()
    reqs_in = [sch.add(p, sp) for p, sp in zip(prompt_ids, sps)]
    recs = {}
    steps = []

    while sch.busy:
        # 混合批次：准入的新请求 prefill + 在途请求 decode 在同一个 forward 里跑
        new = sch.admit()
        new_ids = {r.id for r in new}
        running = [r for r in sch.running if r.id not in new_ids]
        for req in new:
            recs[req.id] = {"first_token_t": None, "finish_t": None,
                            "admit_t": time.perf_counter() - t0, "n_tokens": 0}
        if new or running:
            ts = time.perf_counter()
            next_tok = runner.step_mixed(new, running, 0.0, 1.0)
            stage["mixed"] += time.perf_counter() - ts          # 一个 forward 全含
            for req, token in zip(new + running, next_tok.tolist()):
                before = req.finished
                req.step(token)
                now = time.perf_counter()
                if req in new:
                    recs[req.id]["first_token_t"] = now - t0
                if not before and req.finished:
                    recs[req.id]["finish_t"] = now - t0
            steps.append({"t": time.perf_counter() - t0,
                          "n_running": len(new) + len(running),
                          "n_active": len(new) + len(running)})
        sch.evict()

    reqs = []
    for r in reqs_in:
        rec = recs[r.id]
        rec["n_tokens"] = len(r.generated)
        rec["output"] = list(r.generated)
        reqs.append(rec)
    return {"requests": reqs, "steps": steps, "stage_times": stage,
            "total_time": time.perf_counter() - t0}


# ---------------------------------------------------------------- v2

def run_v2(runner, sch, prompt_ids, sps, batch_size, max_seq_len, eos_ids):
    """v2 paged KV + scheduler 全权管理（schedule/update 接口）。"""
    kv = runner.kv
    stage = {"prefill": 0.0, "decode": 0.0}
    t0 = time.perf_counter()
    reqs_in = [sch.add(p, sp) for p, sp in zip(prompt_ids, sps)]
    recs, steps = {}, []

    while sch.busy:
        batch, is_prefill = sch.schedule()
        if not batch:
            continue
        if is_prefill:
            for req in batch:
                recs[req.id] = {"admit_t": time.perf_counter() - t0,
                                "first_token_t": None, "finish_t": None,
                                "n_tokens": 0, "n_preempts": 0}
        ts = time.perf_counter()
        next_tok = runner.step(batch, is_prefill, 0.0, 1.0)
        stage["prefill" if is_prefill else "decode"] += time.perf_counter() - ts
        now = time.perf_counter()
        for req, token in zip(batch, next_tok.tolist()):
            before = req.finished
            req.step(token)
            if is_prefill:
                recs[req.id]["first_token_t"] = now - t0
            elif not before and req.finished:
                recs[req.id]["finish_t"] = now - t0
        if not is_prefill:
            steps.append({"t": now - t0, "n_running": len(batch),
                          "free_blocks": kv.num_free(), "pool_blocks": kv.max_num_blocks})
        sch.evict()

    reqs = []
    for r in reqs_in:
        rec = recs[r.id]
        rec["n_tokens"] = len(r.generated)
        rec["output"] = list(r.generated)
        reqs.append(rec)
    return {"requests": reqs, "steps": steps, "stage_times": stage,
            "pool_bytes": kv.pool_bytes(), "max_num_blocks": kv.max_num_blocks,
            "total_time": time.perf_counter() - t0}


# ---------------------------------------------------------------- v3

def run_v3(llm, prompt_ids, sps):
    """v3: nano 的 scheduler/block_manager + 我们的模型。循环 = schedule → run → postprocess。"""
    import time as _t
    stage = {"prefill": 0.0, "decode": 0.0}
    t0 = _t.perf_counter()
    seqs_in = [llm.add_request(p, sp) for p, sp in zip(prompt_ids, sps)]
    recs, steps = {}, []

    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        for seq in seqs:
            recs.setdefault(seq.seq_id, {"first_token_t": None, "finish_t": None,
                                         "admit_t": _t.perf_counter() - t0, "n_tokens": 0})
        ts = _t.perf_counter()
        token_ids = llm.model_runner.run(seqs, is_prefill)
        stage["prefill" if is_prefill else "decode"] += _t.perf_counter() - ts
        now = _t.perf_counter()
        for seq in seqs:
            if is_prefill and len(seq.completion_token_ids) > 0 and recs[seq.seq_id]["first_token_t"] is None:
                recs[seq.seq_id]["first_token_t"] = now - t0
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        now = _t.perf_counter()
        for seq in seqs:
            if len(seq.completion_token_ids) > 0 and recs[seq.seq_id]["first_token_t"] is None:
                recs[seq.seq_id]["first_token_t"] = now - t0
            if seq.is_finished and recs[seq.seq_id]["finish_t"] is None:
                recs[seq.seq_id]["finish_t"] = now - t0
        if not is_prefill:
            steps.append({"t": now - t0, "n_running": len(seqs),
                          "n_active": len(seqs)})

    reqs = []
    for seq in seqs_in:
        r = recs[seq.seq_id]
        r["n_tokens"] = len(seq.completion_token_ids)
        r["output"] = list(seq.completion_token_ids)
        reqs.append(r)
    return {"requests": reqs, "steps": steps, "stage_times": stage,
            "total_time": _t.perf_counter() - t0}


# ---------------------------------------------------------------- worker

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["v0", "v1", "v2", "v3"], required=True)
    ap.add_argument("--num-requests", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="覆盖负载：全部请求用同一个 max_tokens")
    ap.add_argument("--gpu-mem-util", type=float, default=0.3,
                    help="v2 池子的显存利用率")
    ap.add_argument("--profile", action="store_true", help="套 torch.profiler 并导出 trace")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, args.engine))

    if args.engine == "v3":
        from nanovllm.engine.llm_engine import LLMEngine
        from samplingParams import SamplingParams
        from transformers import AutoTokenizer
    else:
        from engine.scheduler import Scheduler as S0
        from engine.model_runner import ModelRunner as R0
        from samplingParams import SamplingParams
        from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos_ids = {tok.eos_token_id}
    pad_id = tok.pad_token_id or tok.eos_token_id

    # 负载
    global MAX_TOKEN_TIERS
    if args.max_tokens is not None:
        MAX_TOKEN_TIERS = [args.max_tokens]
    prompt_ids, sps = build_workload_impl(args.num_requests, tok, SamplingParams)

    kwargs = dict(prompt_ids=prompt_ids, sps=sps, batch_size=args.batch_size,
                  max_seq_len=args.max_seq_len)

    def build_runner():
        ekw = {"gpu_memory_utilization": args.gpu_mem_util} if args.engine == "v2" else {}
        return R0(MODEL_PATH, batch_size=args.batch_size, max_seq_len=args.max_seq_len,
                  eos=tok.eos_token_id, **ekw)

    def call():
        if args.engine == "v3":
            llm = LLMEngine(MODEL_PATH, max_num_seqs=args.batch_size,
                            max_model_len=args.max_seq_len,
                            gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True)
            return run_v3(llm, prompt_ids, sps)
        if args.engine == "v0":
            return run_v0(R0, S0, eos_ids=eos_ids, pad_id=pad_id, **kwargs)
        if args.engine == "v1":
            return run_v1(R0, S0, eos_ids=eos_ids, **kwargs)
        runner = build_runner()
        return run_v2(runner, S0(eos_ids=eos_ids, max_seq_len=args.max_seq_len,
                                 max_batch_size=args.batch_size, kv=runner.kv),
                      eos_ids=eos_ids, **kwargs)

    if args.profile:
        import torch
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            result = call()
        trace_path = os.path.join(ROOT, "benchmark", "results",
                                 f"trace_{args.engine}.json.gz")
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        prof.export_chrome_trace(trace_path)
        result["trace"] = trace_path
        result["profiler_table"] = prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=25)
    else:
        result = call()

    result["meta"] = {"engine": args.engine, "num_requests": args.num_requests,
                      "batch_size": args.batch_size, "dtype": "bfloat16"}
    text = json.dumps(result, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"saved to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
