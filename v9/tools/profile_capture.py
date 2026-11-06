"""Capture ONE synthetic text workload through real workers; never a speed test."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

V9 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--pp', type=int, default=1)
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--enable-expert-parallel', action='store_true')
    parser.add_argument('--scenario', choices=('short', 'long', 'mixed'), default='short')
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--prompt-tokens', type=int, default=64)
    parser.add_argument('--long-tokens', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--microbatch-tokens', type=int, default=128)
    parser.add_argument('--max-concurrent-batches', type=int)
    parser.add_argument('--start-message', type=int, default=1)
    parser.add_argument('--messages', type=int, default=0, help='0 = all forwards until shutdown (default)')
    parser.add_argument('--max-kv-blocks', type=int)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--max-num-seqs', type=int, default=128)
    parser.add_argument('--record-shapes', action='store_true')
    args = parser.parse_args()
    if min(args.tp, args.pp, args.dp, args.requests, args.prompt_tokens,
           args.long_tokens, args.max_tokens, args.budget, args.start_message,
           args.microbatch_tokens, args.max_model_len, args.max_num_seqs) < 1:
        parser.error('sizes and limits must be positive')
    if args.messages < 0:
        parser.error('messages must be >= 0 (0 = until shutdown)')
    if args.max_concurrent_batches is not None and args.max_concurrent_batches < 1:
        parser.error('max-concurrent-batches must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new or empty directory; do not mix runs')
    import torch
    expected = args.dp * args.pp * args.tp
    if torch.cuda.device_count() < expected:
        parser.error(f'{expected} workers require {expected} distinct visible GPUs; '
                     f'only {torch.cuda.device_count()} visible. Check CUDA_VISIBLE_DEVICES.')
    args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = []
    for i in range(args.requests):
        length = args.long_tokens if args.scenario == 'long' or (args.scenario == 'mixed' and i % 4 == 0) else args.prompt_tokens
        prefix = tokenizer.encode(f'合成性能分析请求{i}：', add_special_tokens=False)
        unit = tokenizer.encode('请解释推理引擎的调度与通信。', add_special_tokens=False)
        if not unit:
            parser.error('tokenizer produced an empty synthetic token sequence')
        ids = (prefix + unit * (length // len(unit) + 1))[:length]
        rows.append(dict(input_ids=ids, max_tokens=args.max_tokens, arrival_step=0))
    inputs = args.output / 'inputs.json'
    inputs.write_text(json.dumps({args.scenario: rows}, ensure_ascii=False), encoding='utf-8')
    command = [sys.executable, str(V9/'bench.py'), 'offline', '--model', args.model,
        '--output', str(args.output), '--inputs', str(inputs), '--workloads', args.scenario,
        '--tp', str(args.tp), '--pp', str(args.pp), '--dp', str(args.dp),
        '--budget', str(args.budget), '--profile', '--repeats', '1',
        '--max-model-len', str(args.max_model_len), '--max-num-seqs', str(args.max_num_seqs),
        '--microbatch-tokens', str(args.microbatch_tokens),
        '--profile-start-message', str(args.start_message), '--profile-messages', str(args.messages)]
    if args.enable_expert_parallel:
        command.append('--enable-expert-parallel')
    if args.max_concurrent_batches is not None:
        command += ['--max-concurrent-batches', str(args.max_concurrent_batches)]
    if args.max_kv_blocks:
        command += ['--max-kv-blocks', str(args.max_kv_blocks)]
    if args.record_shapes:
        command.append('--record-shapes')
    versions = {}
    for package in ('torch', 'triton', 'transformers', 'safetensors'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    source = {p.relative_to(V9).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in V9.rglob('*.py') if '__pycache__' not in p.parts}
    manifest = dict(command=command, versions=versions, source_sha256=source,
        expected_ranks=args.dp*args.pp*args.tp, model=args.model,
        input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        prompt_lengths=[len(r['input_ids']) for r in rows],
        environment={k: v for k, v in os.environ.items() if k.startswith(('VLLM_LEARN_', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'))},
        scope='synthetic text Thinker only; local worker window; no initialization, tokenizer or HTTP trace; no throughput claims')
    try:
        manifest['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        manifest['nvidia_smi'] = str(error)
    path = args.output/'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.output/'run.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    manifest['exit_code'] = result.returncode
    # The benchmark persists the same semantic inputs with its own formatting.
    # Fingerprint the file the reader will actually find alongside this manifest.
    manifest['input_sha256'] = hashlib.sha256(inputs.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if result.returncode:
        raise SystemExit(f'Capture failed ({result.returncode}); see {args.output}/run.log')
    print(f'Captured {args.output}/trace. Next: profile_analyze.py --input-dir ... --expected-ranks {manifest["expected_ranks"]}')


if __name__ == '__main__':
    main()
