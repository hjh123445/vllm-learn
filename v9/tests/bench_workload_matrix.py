"""Synthetic long/mixed-request matrix with saved per-request timing evidence.

No private dataset or network API is used. DP timestamps are engine replica
completion timestamps, not HTTP streaming latency. Throughput runs are unprofiled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def quantiles(values):
    values = sorted(values)
    return {str(q): values[round((len(values)-1)*q/100)] for q in (50, 95, 99)} if values else {}


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def build_workloads(model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    template = Path(model) / 'chat_template.json'
    if template.exists():
        tok.chat_template = json.loads(template.read_text())['chat_template']

    def prompt(index, target):
        # Keep actual token IDs, including the chat suffix. All long prompts
        # have a unique prefix, so repeated-token APC does not dominate results.
        text = f'实验请求{index}。' + '这是自生成的中文系统性能分析文本。请讨论GPU通信、调度与内存瓶颈。' * (target // 8 + 1)
        ids = tok.encode(text, add_special_tokens=False)
        suffix = tok.encode('\n请用中文概括。', add_special_tokens=False)
        head = tok.encode(f'请求{index}：', add_special_tokens=False)
        overhead = tok.encode(tok.apply_chat_template([{'role': 'user', 'content': ''}],
            tokenize=False, add_generation_prompt=True), add_special_tokens=False)
        # Tokenize complete content once; target lengths below are approximate
        # by design and the actual exact lengths are always persisted.
        content = tok.decode((head + ids[len(head):target-len(overhead)-len(suffix)] + suffix))
        return tok.encode(tok.apply_chat_template([{'role': 'user', 'content': content}],
            tokenize=False, add_generation_prompt=True), add_special_tokens=False)

    specs = {
        'short64': [(48, 32)] * 64,
        'short128': [(48, 32)] * 128,
        'context1k8': [(1024, 32)] * 8,
        'context4k8': [(4096, 32)] * 8,
        'mixed64': [(48 if i % 4 else 2048, 16 if i % 3 else 64) for i in range(64)],
        'arrival64': [(48 if i % 4 else 1024, 16 if i % 3 else 64) for i in range(64)],
    }
    return {name: [dict(input_ids=prompt(i, length), max_tokens=output,
                        arrival_step=(i // 8) * 4 if name == 'arrival64' else 0)
                   for i, (length, output) in enumerate(rows)] for name, rows in specs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--inputs')
    ap.add_argument('--generate-only', action='store_true')
    ap.add_argument('--enable-expert-parallel', action='store_true')
    ap.add_argument('--pp', type=int, default=1)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--dp', type=int, default=1)
    ap.add_argument('--budget', type=int, default=512)
    ap.add_argument('--microbatch-tokens', type=int, default=128)
    ap.add_argument('--lanes', type=int, default=2)
    ap.add_argument('--mode', choices=['baseline', 'optimized'], default='optimized')
    ap.add_argument('--workloads', default='short64,short128,context1k8,context4k8,mixed64,arrival64')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--max-kv-blocks', type=int, default=None)
    ap.add_argument('--ablate', action='store_true')
    ap.add_argument('--profile', action='store_true',
                    help='capture only; reported timings are diagnostic, no benchmark percentiles/rates')
    ap.add_argument('--profile-start-message', type=int, default=3)
    ap.add_argument('--profile-messages', type=int, default=16)
    args = ap.parse_args()
    selected = args.workloads.split(',')
    if args.profile:
        if len(selected) != 1 or args.ablate:
            ap.error('--profile requires one --workloads name and no --ablate')
        args.repeats = 1
    if min(args.tp, args.dp, args.pp, args.repeats, args.budget, args.microbatch_tokens) < 1:
        ap.error('parallel sizes, repeats and token budgets must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(Path(args.inputs).read_text()) if args.inputs else build_workloads(args.model)
    write(out / 'inputs.json', inputs)
    if args.generate_only:
        print({name: dict(requests=len(rows), prompt_lengths=sorted(set(len(r['input_ids']) for r in rows)))
               for name, rows in inputs.items()}, flush=True)
        return
    from samplingParams import SamplingParams
    from utils.observed_runner import ObservedModelRunner, ObservedPipelineTransport
    from utils.observation_config import configure_observation
    configure_observation(out, profile=args.profile, start_message=args.profile_start_message,
                          messages=args.profile_messages)
    config = dict(max_num_seqs=128, max_model_len=8192, max_num_batched_tokens=args.budget,
                  max_kvcache_blocks=args.max_kv_blocks,
                  gpu_memory_utilization=.70, enforce_eager=True, enable_prefix_caching=False)
    kw = dict(tp_size=args.tp, enable_expert_parallel=args.enable_expert_parallel, pp_size=args.pp, master_port=31030,
              runner_cls=ObservedModelRunner, transport_cls=ObservedPipelineTransport,
              pipeline_microbatch_tokens=args.microbatch_tokens, pipeline_lanes=args.lanes,
              pipeline_mode=args.mode, config_kwargs=config)
    write(out / 'config.json', dict(vars(args), engine_config=config,
          ep_derived_from_dp_tp=args.enable_expert_parallel,
          numerical_policy={name: os.getenv(name, default) for name, default in (
              ('VLLM_LEARN_EXPERT_BACKEND', 'loop'),
              ('VLLM_LEARN_EP_WEIGHT_AT_SOURCE', '0'),
              ('VLLM_LEARN_ROUTER_LOGITS_DTYPE', 'model'),
              ('VLLM_LEARN_REFERENCE_MATH', '1'),
              ('VLLM_LEARN_MOE_REDUCTION', 'fp32'),
              ('VLLM_LEARN_ATTENTION_BACKEND', 'default'))},
          source_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):
              hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                  Path(__file__).resolve().parents[1] / 'layers/moe.py',
                  Path(__file__).resolve().parents[1] / 'layers/parallel.py',
                  Path(__file__).resolve().parents[1] / 'engine/core.py',
                  Path(__file__).resolve().parents[1] / 'engine/worker.py')},
          measurement=('diagnostic profile; exclude all timings from performance tables' if args.profile
                       else 'local engine completion timestamps, excluding initialization; no HTTP'),
          inputs_sha256=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()))
    start = time.perf_counter()
    if args.dp > 1:
        from engine.frontend import DPFrontend
        core = DPFrontend(args.model, dp_size=args.dp, **kw)
    else:
        from engine.core import EngineCore
        core = EngineCore(args.model, **kw)
    construction_s = time.perf_counter() - start
    results = []

    def run(name, rows, repeat, policy):
        if args.dp == 1:
            core.pipeline_microbatch_tokens, core.pipeline_lanes, core.pipeline_mode = policy
        started = time.perf_counter()
        pending = list(enumerate(rows))
        handles, arrivals, admissions, outputs, step_times = {}, {}, {}, {}, []
        nstep = 0
        while pending or (not core.is_finished() if args.dp == 1 else len(outputs) < len(handles)):
            while pending and pending[0][1].get('arrival_step', 0) <= nstep:
                index, row = pending.pop(0)
                sp = SamplingParams(temperature=0., max_tokens=row['max_tokens'], ignore_eos=True)
                seq = core.add_request(row['input_ids'], sp)
                if args.dp > 1:
                    seq = seq[0]
                handles[seq.seq_id] = (index, seq)
                arrivals[seq.seq_id] = []
                admissions[seq.seq_id] = time.perf_counter()
            before = time.perf_counter()
            completed = core.step_all() if args.dp > 1 else dict(core.step())
            step_times.append(time.perf_counter() - before)
            for event in core.last_step_events:
                if event['valid_decode_token']:
                    arrivals[event['seq_id']].append(event['step_end'])
            outputs.update(completed)
            nstep += 1
            if nstep > 5000:
                raise RuntimeError('Scheduler made no bounded progress')
        elapsed = time.perf_counter() - started
        request_rows = []
        for seq_id, (index, seq) in handles.items():
            ids = outputs[seq_id]
            times = arrivals[seq_id]
            if len(ids) != rows[index]['max_tokens'] or len(times) != len(ids):
                raise AssertionError((index, len(ids), len(times), rows[index]['max_tokens']))
            request_rows.append(dict(index=index, seq_id=seq_id,
                prompt_tokens=len(rows[index]['input_ids']), output_tokens=len(ids), output_ids=ids,
                admission_s=admissions[seq_id]-started,
                arrival_s=[t-started for t in times], ttft_s=times[0]-admissions[seq_id],
                e2e_s=times[-1]-admissions[seq_id],
                tpot_s=(times[-1]-times[0])/(len(times)-1) if len(times)>1 else None))
        request_rows.sort(key=lambda r:r['index'])
        record = dict(workload=name, repeat=repeat, policy=dict(microbatch_tokens=policy[0],
            lanes=policy[1], mode=policy[2]), dp=args.dp, pp=args.pp, ep=(args.dp * args.tp if args.enable_expert_parallel else 1), tp=args.tp,
            wall_s=elapsed, construction_s=construction_s, profiled=args.profile,
            benchmark_valid=not args.profile,
            output_tokens=sum(r['output_tokens'] for r in request_rows),
            output_tok_s=None if args.profile else sum(r['output_tokens'] for r in request_rows)/elapsed,
            request_s=None if args.profile else len(request_rows)/elapsed,
            ttft_s=None if args.profile else quantiles([r['ttft_s'] for r in request_rows]),
            e2e_s=None if args.profile else quantiles([r['e2e_s'] for r in request_rows]),
            tpot_s=None if args.profile else quantiles([r['tpot_s'] for r in request_rows if r['tpot_s'] is not None]),
            itl_s=None if args.profile else quantiles([b-a for r in request_rows
                for a,b in zip(r['arrival_s'], r['arrival_s'][1:])]),
            percentile_method='nearest observed order statistic: round((n-1)*q/100); units seconds',
            step_s=step_times, requests=request_rows)
        results.append(record)
        write(out/'results.json', results)
        print(json.dumps({k:v for k,v in record.items() if k not in ('requests','step_s')}), flush=True)

    try:
        # Profile the selected workload itself. Capturing a preceding request
        # warmup would exhaust a bounded window before the intended input.
        if not args.profile:
            warmup = inputs.get('short64', inputs[selected[0]])
            run('warmup', [dict(input_ids=r['input_ids'], max_tokens=4) for r in warmup[:8]],
                -1, (args.microbatch_tokens, args.lanes, args.mode))
        policies = [(args.microbatch_tokens, args.lanes, args.mode)]
        if args.ablate:
            if args.pp <= 1 or args.dp != 1 or args.lanes != 2:
                raise ValueError('PP ablation requires pp>1, dp=1, transport lanes=2')
            policies = [(budget, lanes, mode) for budget in (32,128,512)
                        for lanes in (1,2) for mode in ('baseline','optimized')]
        for name in args.workloads.split(','):
            for repeat in range(args.repeats):
                for policy in (policies if repeat%2==0 else list(reversed(policies))):
                    run(name, inputs[name], repeat, policy)
        checks = []
        for name in args.workloads.split(','):
            rows = [r for r in results if r['workload']==name]
            # Includes policy parity, not only repeatability.
            reference = [r['output_ids'] for r in rows[0]['requests']]
            checks.append(dict(workload=name, passed=all(
                [r['output_ids'] for r in row['requests']]==reference for row in rows),
                runs=len(rows)))
        write(out/'correctness.json', dict(scope='same topology greedy repeat and policy parity; not HF parity',
                                          checks=checks, passed=all(c['passed'] for c in checks)))
        if not all(c['passed'] for c in checks):
            raise RuntimeError('Output divergence; see correctness.json')
    finally:
        core.shutdown()


if __name__ == '__main__':
    main()
