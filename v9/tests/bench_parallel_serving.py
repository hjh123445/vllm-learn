"""Reproducible synthetic-text benchmark; no server datasets are read.

Run EP8 and PP2xEP4 sequentially on the same eight GPUs. Profiler runs are
separate from throughput runs. Output-token throughput excludes initialization.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer
from engine.core import EngineCore
from utils.observed_runner import ObservedModelRunner, ObservedPipelineTransport
from samplingParams import SamplingParams


def quantiles(xs):
    xs = sorted(xs)
    return {str(q): xs[round((len(xs) - 1) * q / 100)] for q in (50, 95, 99)} if xs else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--enable-expert-parallel', action='store_true')
    ap.add_argument('--pp', type=int, default=1)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--output', required=True)
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--verify', action='store_true',
                    help='Check repeated greedy outputs, including PP mode parity')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--trace-mode', choices=['baseline', 'optimized'], default='optimized')
    ap.add_argument('--trace-workload', choices=['short8', 'long16'], default='short8')
    ap.add_argument('--include-large', action='store_true', help='Add a 64-request decode workload')
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    from utils.observation_config import configure_observation
    configure_observation(out, profile=args.trace, start_message=3, messages=4)
    tok = AutoTokenizer.from_pretrained(args.model)
    template = Path(args.model) / 'chat_template.json'
    if template.exists():
        tok.chat_template = json.loads(template.read_text())['chat_template']

    def encode(text):
        return tok.encode(tok.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=False,
            add_generation_prompt=True), add_special_tokens=False)

    short = [encode(f'问题{i}：用一句中文解释专家并行和流水线并行的区别。') for i in range(8)]
    long = [encode(f'问题{i}：' + ('这是公开合成的性能测试文本，讨论GPU通信与推理调度。' * 12)
                   + '请概括上述内容。') for i in range(16)]
    workload = {'short8': short, 'long16': long}
    if args.include_large:
        workload['short64'] = short * 8
    (out / 'inputs.json').write_text(json.dumps(workload, ensure_ascii=False), encoding='utf-8')
    config = dict(max_num_seqs=64 if args.include_large else 32, max_model_len=1024,
                  max_num_batched_tokens=512, gpu_memory_utilization=0.70,
                  enforce_eager=True)
    started = time.perf_counter()
    core = EngineCore(args.model, tp_size=args.tp, enable_expert_parallel=args.enable_expert_parallel,
                      runner_cls=ObservedModelRunner, transport_cls=ObservedPipelineTransport,
                      pp_size=args.pp, gpus=list(range(args.tp * args.pp)),
                      master_port=30873, pipeline_microbatch_tokens=32,
                      pipeline_lanes=2, config_kwargs=config)
    init_s = time.perf_counter() - started
    results = []

    def run(name, inputs, mode, repeat, max_tokens=16):
        core.pipeline_mode = mode
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        start = time.perf_counter()
        seqs = [core.add_request(ids, sp) for ids in inputs]
        arrivals = {s.seq_id: [] for s in seqs}
        steps = []
        while not core.is_finished():
            t = time.perf_counter()
            core.step()
            end = time.perf_counter()
            steps.append(end - t)
            for event in core.last_step_events:
                if event['valid_decode_token']:
                    arrivals[event['seq_id']].append(end - start)
        elapsed = time.perf_counter() - start
        outputs = [list(s.completion_token_ids) for s in seqs]
        ttft = [times[0] for times in arrivals.values() if times]
        itl = [b-a for times in arrivals.values() for a,b in zip(times, times[1:])]
        record = dict(workload=name, mode=mode, repeat=repeat,
                      tp=args.tp, ep=(args.tp if args.enable_expert_parallel else 1), pp=args.pp, init_s=init_s,
                      wall_s=elapsed, output_tokens=sum(map(len, outputs)),
                      output_tok_s=None if args.trace else sum(map(len, outputs))/elapsed,
                      ttft_s=None if args.trace else quantiles(ttft),
                      itl_s=None if args.trace else quantiles(itl),
                      benchmark_valid=not args.trace,
                      step_s=steps, prompt_lengths=list(map(len, inputs)),
                      output_ids=outputs, texts=[tok.decode(ids) for ids in outputs],
                      measurement='engine step completion timestamps; not client streaming',
                      input_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
                      profiled=args.trace)
        results.append(record)
        (out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k not in ('output_ids','texts','step_s')}, ensure_ascii=False), flush=True)

    try:
        run('warmup', workload[args.trace_workload], args.trace_mode, -1, max_tokens=8)
        if args.verify:
            checks = []
            for name, inputs in workload.items():
                modes = ('baseline', 'optimized') if args.pp > 1 else ('optimized',)
                runs = []
                for mode in modes:
                    for repeat in range(2):
                        run(name, inputs, mode, repeat, max_tokens=8)
                        runs.append(results[-1])
                ref = runs[0]['output_ids']
                for rec in runs[1:]:
                    mismatch = [i for i, (a, b) in enumerate(zip(ref, rec['output_ids'])) if a != b]
                    checks.append(dict(workload=name, mode=rec['mode'],
                                       repeat=rec['repeat'], mismatched_requests=mismatch,
                                       passed=not mismatch))
            passed = all(c['passed'] for c in checks)
            (out / 'correctness.json').write_text(json.dumps(
                dict(passed=passed, checks=checks,
                     scope='Same batch greedy repeatability and PP mode parity; not HF quality equivalence'),
                ensure_ascii=False, indent=2), encoding='utf-8')
            if not passed:
                raise RuntimeError('Greedy consistency failed; see correctness.json')
            return
        if args.trace:
            return
        for name, inputs in workload.items():
            for repeat in range(args.repeats):
                modes = ('optimized', 'baseline') if args.pp > 1 else ('optimized',)
                if repeat % 2:
                    modes = tuple(reversed(modes))
                for mode in modes:
                    run(name, inputs, mode, repeat)
        checks = []
        for name in workload:
            modes = ("optimized", "baseline") if args.pp > 1 else ("optimized",)
            for mode in modes:
                rows = [r for r in results
                        if r["workload"] == name and r["mode"] == mode]
                checks.append({"workload": name, "mode": mode,
                               "repeats": len(rows),
                               "passed": bool(rows) and all(
                                   r["output_ids"] == rows[0]["output_ids"]
                                   for r in rows)})
        (out / "correctness.json").write_text(json.dumps(
            {"passed": all(row["passed"] for row in checks),
             "scope": "Same-input repeated greedy output equality per mode; not HF quality parity",
             "checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8")
        if not all(row["passed"] for row in checks):
            raise RuntimeError("Greedy repeatability failed; see correctness.json")
    finally:
        core.shutdown()
    del core
    gc.collect()


if __name__ == '__main__':
    main()
