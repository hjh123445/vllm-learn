"""Check actual cached multi-token generation against saved HF continuations.

Fixed-prefix probes isolate one-token math, but do not exercise cached decode.
This diagnostic complements them without claiming task-level model quality.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--enable-expert-parallel', action='store_true')
    parser.add_argument('--pp', type=int, default=1)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--token-budget', type=int, default=512)
    parser.add_argument('--microbatch-tokens', type=int, default=128)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.70)
    parser.add_argument('--max-kv-blocks', type=int, default=None)
    parser.add_argument('--tp', type=int, default=1)
    args = parser.parse_args()
    from engine.core import EngineCore
    from utils.observed_runner import ObservedModelRunner
    from samplingParams import SamplingParams

    reference = json.loads(Path(args.reference).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    core = EngineCore(args.model, tp_size=args.tp, enable_expert_parallel=args.enable_expert_parallel, pp_size=args.pp, master_port=31030,
        runner_cls=ObservedModelRunner,
        pipeline_microbatch_tokens=args.microbatch_tokens, pipeline_lanes=2, config_kwargs=dict(
            max_model_len=8192, max_num_batched_tokens=args.token_budget, max_num_seqs=128,
            gpu_memory_utilization=args.gpu_memory_utilization, max_kvcache_blocks=args.max_kv_blocks,
            enforce_eager=True, enable_prefix_caching=False))
    handles = {}
    completed = {}

    def submit(index):
        item = reference[index]
        seq = core.add_request(item['input_ids'], SamplingParams(temperature=0.,
            max_tokens=len(item['output_ids']), ignore_eos=True))
        handles[index] = seq.seq_id

    def drain():
        steps = 0
        while not core.is_finished():
            completed.update(dict(core.step()))
            steps += 1
            if steps > 5000:
                raise RuntimeError('No bounded progress during generation')

    try:
        if args.batch:
            for i in range(len(reference)):
                submit(i)
            drain()
        else:
            for i in range(len(reference)):
                submit(i)
                drain()
        records = []
        for i, item in enumerate(reference):
            actual, expected = completed[handles[i]], item['output_ids']
            if len(actual) != len(expected):
                raise AssertionError('Incomplete output')
            first = next((j for j, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
            records.append(dict(workload=item['workload'], index=item['index'],
                input_tokens=len(item['input_ids']), actual_output_ids=actual,
                reference_output_ids=expected, exact_match=first is None,
                first_divergence=first, matching_prefix_tokens=first if first is not None else len(actual)))
        result = dict(cases=len(records), exact_matches=sum(r['exact_match'] for r in records),
            passed=all(r['exact_match'] for r in records), records=records,
            scope='Synthetic HF continuation parity including cached decode; not a quality benchmark',
            caution='After first divergence, later positions have different input prefixes')
        (out/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (out/'config.json').write_text(json.dumps(dict(vars(args), numerical_policy={
            name: value for name, value in os.environ.items() if name.startswith('VLLM_LEARN_')},
            reference_backend='HF explicit grouped_mm and SDPA'), indent=2))
        print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)
    finally:
        core.shutdown()


if __name__ == '__main__':
    main()
