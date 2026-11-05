"""DP2 x TP4 -> EP8 benchmark with globally unique request ids and saved outputs."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.frontend import DPFrontend
from samplingParams import SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(Path(args.inputs).read_text())
    inputs['short64'] = inputs['short8'] * 8
    (out / 'inputs.json').write_text(json.dumps(inputs))
    start = time.perf_counter()
    fe = DPFrontend(args.model, dp_size=2, tp_size=4, enable_expert_parallel=True,
                    master_port=30910,
                    config_kwargs=dict(max_num_seqs=64, max_model_len=1024,
                                       max_num_batched_tokens=512,
                                       gpu_memory_utilization=0.70, enforce_eager=True))
    results = []

    def run(name, batch, repeat, n=16):
        started = time.perf_counter()
        handles = [fe.add_request(ids, SamplingParams(temperature=0.0,
                   max_tokens=n, ignore_eos=True))[0] for ids in batch]
        outputs = fe.run()
        elapsed = time.perf_counter() - started
        ordered = [outputs[s.seq_id] for s in handles]
        assert len(outputs) == len(batch), (len(outputs), len(batch))
        assert all(len(ids) == n for ids in ordered)
        record = dict(workload=name, repeat=repeat, dp=2, ep=4, pp=1,
                      wall_s=elapsed, output_tokens=sum(map(len, ordered)),
                      output_tok_s=sum(map(len, ordered))/elapsed,
                      output_ids=ordered, profiled=False)
        results.append(record)
        (out / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({k:v for k,v in record.items() if k!='output_ids'}), flush=True)

    try:
        run('warmup', inputs['short8'], -1, n=8)
        (out / 'initialization.json').write_text(json.dumps(
            {'construction_and_first_warmup_s': time.perf_counter()-start}))
        for name, batch in inputs.items():
            for repeat in range(3):
                run(name, batch, repeat)
        checks = []
        for name in inputs:
            rows = [r for r in results if r['workload']==name]
            checks.append(dict(workload=name, passed=all(
                r['output_ids']==rows[0]['output_ids'] for r in rows)))
        (out / 'correctness.json').write_text(json.dumps(checks, indent=2))
        if not all(c['passed'] for c in checks):
            raise RuntimeError('DP repeatability failed')
    finally:
        fe.shutdown()


if __name__ == '__main__':
    main()
