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


def build_mm_workload(model, mm_jsonl, mm_pixels_out):
    """多模态负载:jsonl 行 {"text": str, "image": path, "max_tokens": int}。
    展开 <image> 为 N 个 image_pad token(N=t*(h/2)*(w/2)),像素存 npz 供引擎侧按需取。"""
    import numpy as np
    from transformers import AutoProcessor
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from layers.mm_preprocess import image_to_patches, mrope_positions
    proc = AutoProcessor.from_pretrained(model)
    rows, pixels, pos3s = [], [], []
    for line in Path(mm_jsonl).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        media = r.get('images') or ([r['image']] if r.get('image') else [])
        if not media:
            raise ValueError('multimodal row requires image or images')
        conv = [{'role': 'user', 'content': ([{'type': 'image'} for _ in media]
                 + [{'type': 'text', 'text': r['text']}])}]
        text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        pad = '<|image_pad|>'
        enc = proc.tokenizer
        image_id = enc.convert_tokens_to_ids(pad)
        if image_id is None or image_id == enc.unk_token_id:
            raise ValueError('tokenizer does not define image_pad')
        parts = text.split(pad)
        if len(parts) != len(media) + 1:
            raise ValueError('image count disagrees with chat template placeholders')
        ids, image_rows, spans = [], [], []
        for i, path in enumerate(media):
            ids.extend(enc.encode(parts[i], add_special_tokens=False))
            image_path = Path(path)
            if not image_path.is_absolute():
                image_path = Path(mm_jsonl).resolve().parent / image_path
            pat, grid = image_to_patches(str(image_path))
            t, gh, gw = grid[0].tolist()
            pos = len(ids)
            ids.extend([image_id] * (t * (gh // 2) * (gw // 2)))
            image_rows.append(dict(pos=pos, grid=[t, gh, gw], pixel_idx=len(pixels),
                source_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest()))
            spans.append((pos, t, gh // 2, gw // 2))
            pixels.append(pat.numpy())
        ids.extend(enc.encode(parts[-1], add_special_tokens=False))
        pos3, delta = mrope_positions(len(ids), spans)
        pos3s.append(pos3)
        rows.append(dict(input_ids=ids, max_tokens=int(r.get('max_tokens', 16)),
                         mm={'images': image_rows, 'delta': delta, 'pos3_idx': len(pos3s) - 1}))
    save = {f'p{i}': p for i, p in enumerate(pixels)}
    save.update({f'pos3_{i}': p for i, p in enumerate(pos3s)})
    np.savez_compressed(mm_pixels_out, **save)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--mm-jsonl', help='multimodal workload jsonl (enables VLLM_LEARN_MM path)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--inputs')
    ap.add_argument('--generate-only', action='store_true')
    ap.add_argument('--enable-expert-parallel', action='store_true')
    ap.add_argument('--pp', type=int, default=1)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--dp', type=int, default=1)
    ap.add_argument('--budget', type=int, default=512)
    ap.add_argument('--microbatch-tokens', type=int, default=128)
    ap.add_argument('--max-concurrent-batches', type=int, help='in-flight PP batches; default: pp size')
    ap.add_argument('--workloads', default='short64,short128,context1k8,context4k8,mixed64,arrival64')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--max-kv-blocks', type=int, default=None)
    ap.add_argument('--max-model-len', type=int, default=8192)
    ap.add_argument('--max-num-seqs', '--max-batch-size', type=int, default=128)
    ap.add_argument('--profile', action='store_true',
                    help='capture only; reported timings are diagnostic, no benchmark percentiles/rates')
    ap.add_argument('--decode-graph', action='store_true', help='static decode preset with strict batch-invariant numerics: dense EP/BMM, Triton attention and QK; requires DP=PP=1 and TP>1')
    ap.add_argument('--batch-invariant', action='store_true',
                    help='fixed GEMM tiles, RMSNorm and rank/expert reduction order; costs performance')
    ap.add_argument('--profile-start-message', type=int, default=1)
    ap.add_argument('--profile-messages', type=int, default=0, help='0 = all forwards until shutdown (default)')
    ap.add_argument('--record-shapes', action='store_true')
    args = ap.parse_args()
    if args.batch_invariant:
        os.environ['VLLM_LEARN_BATCH_INVARIANT'] = '1'
    if args.decode_graph:
        if args.dp != 1 or args.pp != 1 or args.tp <= 1 or not args.enable_expert_parallel:
            ap.error('--decode-graph requires DP=PP=1, TP>1 and --enable-expert-parallel')
        os.environ.update(VLLM_LEARN_DECODE_GRAPH='1', VLLM_LEARN_DECODE_GRAPH_STRICT='1',
            VLLM_LEARN_BATCH_INVARIANT='1',
            VLLM_LEARN_EP_DENSE_DECODE='1',
            VLLM_LEARN_DENSE_EXPERT_BACKEND='bmm', VLLM_LEARN_REFERENCE_QK='0',
            VLLM_LEARN_ATTENTION_BACKEND='triton')
    if min(args.tp, args.dp, args.pp, args.repeats, args.budget, args.microbatch_tokens) < 1:
        ap.error('parallel sizes, repeats and token budgets must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(Path(args.inputs).read_text()) if args.inputs else build_workloads(args.model)
    if args.inputs and Path(args.inputs).resolve().parent != out.resolve():
        source_pixels = Path(args.inputs).resolve().parent / 'mm_pixels.npz'
        if source_pixels.exists():
            import shutil
            shutil.copyfile(source_pixels, out / 'mm_pixels.npz')
    if args.mm_jsonl:
        os.environ['VLLM_LEARN_MM'] = '1'
        mm_rows = build_mm_workload(args.model, args.mm_jsonl, out / 'mm_pixels.npz')
        inputs = dict(inputs)
        inputs['mm'] = mm_rows
    selected = args.workloads.split(',')
    if any(name not in inputs for name in selected):
        ap.error('every selected workload must exist in the inputs')
    write(out / 'inputs.json', inputs)
    if args.generate_only:
        print({name: dict(requests=len(rows), prompt_lengths=sorted(set(len(r['input_ids']) for r in rows)))
               for name, rows in inputs.items()}, flush=True)
        return
    from samplingParams import SamplingParams
    from utils.observed_runner import ObservedModelRunner, ObservedPipelineTransport
    from utils.observation_config import configure_observation
    configure_observation(out, profile=args.profile, start_message=args.profile_start_message,
                          messages=args.profile_messages, record_shapes=args.record_shapes)
    config = dict(max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len, max_num_batched_tokens=args.budget,
                  max_kvcache_blocks=args.max_kv_blocks,
                  gpu_memory_utilization=float(__import__("os").getenv("VLLM_BENCH_UTIL", ".70")),
                  enable_prefix_caching=False)
    kw = dict(tp_size=args.tp, enable_expert_parallel=args.enable_expert_parallel, pp_size=args.pp, master_port=31030,
              runner_cls=ObservedModelRunner, transport_cls=ObservedPipelineTransport,
              pipeline_microbatch_tokens=args.microbatch_tokens,
              max_concurrent_batches=args.max_concurrent_batches, config_kwargs=config)
    write(out / 'config.json', dict(vars(args), engine_config=config,
          ep_derived_from_dp_tp=args.enable_expert_parallel,
          numerical_policy={name: os.getenv(name, default) for name, default in (
              ('VLLM_LEARN_EXPERT_BACKEND', 'loop'),
              ('VLLM_LEARN_EP_WEIGHT_AT_SOURCE', '0'),
              ('VLLM_LEARN_ROUTER_LOGITS_DTYPE', 'model'),
              ('VLLM_LEARN_REFERENCE_MATH', '1'),
              ('VLLM_LEARN_MOE_REDUCTION', 'fp32'),
              ('VLLM_LEARN_ATTENTION_BACKEND', 'sdpa_reference'),
              ('VLLM_LEARN_REFERENCE_QK', os.getenv('VLLM_LEARN_REFERENCE_MATH', '1')),
              ('VLLM_LEARN_EP_DENSE_DECODE', '0'),
              ('VLLM_LEARN_DENSE_EXPERT_BACKEND', 'bmm'),
              ('VLLM_LEARN_DECODE_GRAPH', '0'),
              ('VLLM_LEARN_DECODE_GRAPH_STRICT', '1'), ('VLLM_LEARN_MM', '0'),
              ('VLLM_LEARN_BATCH_INVARIANT', '0'))},
          tp_reduction_dtype=os.getenv('VLLM_LEARN_TP_REDUCTION_DTYPE', 'model'),
          source_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):
              hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(Path(__file__).resolve().parents[1].rglob('*.py'))
              if '__pycache__' not in path.parts},
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

    def run(name, rows, repeat):
        started = time.perf_counter()
        pending = list(enumerate(rows))
        handles, arrivals, admissions, outputs, step_times = {}, {}, {}, {}, []
        nstep = 0
        while pending or (not core.is_finished() if args.dp == 1 else len(outputs) < len(handles)):
            while pending and pending[0][1].get('arrival_step', 0) <= nstep:
                index, row = pending.pop(0)
                sp = SamplingParams(temperature=0., max_tokens=row['max_tokens'], ignore_eos=True)
                mm = None
                if row.get('mm') is not None:
                    import numpy as _np
                    npz = _np.load(out / 'mm_pixels.npz')
                    m = row['mm']
                    images = m.get('images', [m])  # old single-image input files remain readable
                    mm = {'images': [{'pixels': npz[f"p{im['pixel_idx']}"], 'grid': im['grid'],
                                      'pos': im['pos']} for im in images],
                          'delta': m['delta'], 'pos3': npz[f"pos3_{m['pos3_idx']}"]}
                    npz.close()
                seq = core.add_request(row['input_ids'], sp, mm=mm)
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
            expect = min(rows[index]['max_tokens'], seq.max_tokens)
            if len(ids) != expect or len(times) != len(ids):
                raise AssertionError((index, len(ids), len(times), rows[index]['max_tokens']))
            request_rows.append(dict(index=index, seq_id=seq_id,
                prompt_tokens=len(rows[index]['input_ids']), output_tokens=len(ids), output_ids=ids,
                admission_s=admissions[seq_id]-started,
                arrival_s=[t-started for t in times], ttft_s=times[0]-admissions[seq_id],
                e2e_s=times[-1]-admissions[seq_id],
                tpot_s=(times[-1]-times[0])/(len(times)-1) if len(times)>1 else None))
        request_rows.sort(key=lambda r:r['index'])
        record = dict(workload=name, repeat=repeat, policy=dict(microbatch_tokens=args.microbatch_tokens,
            max_concurrent_batches=args.max_concurrent_batches or args.pp), dp=args.dp, pp=args.pp, ep=(args.dp * args.tp if args.enable_expert_parallel else 1), tp=args.tp,
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
            run('warmup', [dict(input_ids=r['input_ids'], max_tokens=4) for r in warmup[:8]], -1)
        for name in selected:
            for repeat in range(args.repeats):
                run(name, inputs[name], repeat)
        checks = []
        for name in selected:
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
    if args.profile:
        from utils.trace_audit import finish_profile
        finish_profile(out, expected_ranks=args.dp * args.pp * args.tp)


if __name__ == '__main__':
    main()
