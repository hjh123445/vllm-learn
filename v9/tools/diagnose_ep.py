"""Capture the ORIGINAL cached decode, outside production model/engine code.

Numerical diagnostics only: hooks synchronize CUDA and invalidate timings.
Use an explicitly supplied synthetic checkpoint; see docs/精度排查.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.model_runner import ModelRunner
from layers.parallel import STATE


def copy_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return tuple(copy_value(item) for item in value)
    return value


class DiagnosticRunner(ModelRunner):
    """Dependency-injected runner; warmup is deliberately excluded."""

    def __init__(self, config, dtype=None):
        self.record = None
        self.index = 0
        self.handles = []
        self.history = {}
        torch.backends.cuda.matmul.allow_tf32 = False
        super().__init__(config, getattr(torch, os.environ['EP_DIAG_DTYPE']))
        self.directory = Path(os.environ['EP_DIAG_OUTPUT']) / (
            f'dp{STATE.replica_rank}_tp{STATE.tp_rank}')
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, module in self.model.named_modules():
            if (name in ('model.embed_tokens', 'lm_head') or
                    name.startswith('model.layers.') and
                    (name.count('.') == 2 or name.endswith(('.self_attn', '.mlp', '.router')))):
                self.handles.append(module.register_forward_pre_hook(self.before(name)))
                self.handles.append(module.register_forward_hook(self.after(name)))
        for i, layer in enumerate(self.model.model.layers):
            experts = layer.mlp.experts
            torch.save(dict(w13=copy_value(experts.w13), w2=copy_value(experts.w2),
                            e_start=experts.e_start, i_start=experts.i_start),
                       self.directory / f'weights_layer{i}.pt')
            original = experts.compute_sorted

            def compute(xs, offsets, original=original, i=i):
                result = original(xs, offsets)
                if self.record is not None:
                    self.record['experts'][str(i)] = copy_value((xs, offsets, result))
                return result

            experts.compute_sorted = compute
        self.handles.append(self.sampler.register_forward_pre_hook(self.before('sampler')))
        self.handles.append(self.sampler.register_forward_hook(self.after('sampler')))

    def before(self, name):
        def hook(_module, args):
            if self.record is not None:
                self.record['modules'].setdefault(name, {})['input'] = copy_value(args)
        return hook

    def after(self, name):
        def hook(_module, _args, output):
            if self.record is not None:
                self.record['modules'].setdefault(name, {})['output'] = copy_value(output)
        return hook

    def run(self, seqs, is_prefill=True, microbatch_id=0):
        if not hasattr(self, 'directory'):
            return super().run(seqs, is_prefill, microbatch_id=microbatch_id)
        sequences = []
        for s in seqs:
            history = self.history.setdefault(s.seq_id, [])
            window = s._tokens if hasattr(s, '_tokens') else s.token_ids[s.num_cached_tokens:]
            base = s.num_cached_tokens if is_prefill else len(s) - 1
            if len(history) < base:
                raise RuntimeError('Diagnostic input history has a gap')
            history[base:base + len(window)] = window
            sequences.append(dict(seq_id=s.seq_id, token_ids=list(history),
                seq_len=len(s), cached=s.num_cached_tokens,
                scheduled=s.num_scheduled_tokens, block_table=list(s.block_table)))
        self.record = dict(index=self.index, is_prefill=is_prefill,
            microbatch_id=microbatch_id, dp_rank=STATE.replica_rank,
            tp_rank=STATE.tp_rank, ep_rank=STATE.ep_rank,
            sequences=sequences,
            modules={}, experts={})
        try:
            result = super().run(seqs, is_prefill, microbatch_id=microbatch_id)
            self.record['result'] = copy_value(result)
            return result
        finally:
            torch.save(self.record, self.directory / f'call{self.index:03d}.pt')
            self.record = None
            self.index += 1

    def shutdown(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        super().shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ep', action='store_true')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--expert-precision', choices=['model', 'float32'], default='model')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def sha256(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    source = Path(__file__).resolve().parents[1]
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        source_sha256={p.relative_to(source).as_posix(): sha256(p)
                       for p in sorted(source.rglob('*.py')) if '__pycache__' not in p.parts},
        checkpoint_sha256={p.name: sha256(p) for p in sorted(Path(args.model).glob('*'))
                           if p.is_file() and p.suffix in ('.json', '.safetensors')},
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}), indent=2))
    os.environ.update(EP_DIAG_DTYPE=args.dtype, EP_DIAG_OUTPUT=str(args.output.resolve()),
        VLLM_LEARN_REFERENCE_MATH='1', VLLM_LEARN_ATTENTION_BACKEND='sdpa_reference',
        VLLM_LEARN_EXPERT_BACKEND='loop', VLLM_LEARN_EXPERT_PRECISION=args.expert_precision,
        VLLM_LEARN_WORKER_TIMEOUT_S='90')
    from engine.frontend import DPFrontend
    from samplingParams import SamplingParams
    prompts = [[1, 2], list(range(2, 12)), list(range(3, 28)), [5, 6, 7], [1, 2]]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    frontend = DPFrontend(args.model, dp_size=2, tp_size=2,
        enable_expert_parallel=args.ep, runner_cls=DiagnosticRunner,
        master_port=port, gpus=list(range(4)), pipeline_microbatch_tokens=8,
        config_kwargs=dict(max_model_len=128, max_num_batched_tokens=8, max_num_seqs=16,
                           gpu_memory_utilization=.98, max_kvcache_blocks=16))
    try:
        handles = [frontend.add_request(p, SamplingParams(temperature=0., max_tokens=4,
                    ignore_eos=True))[0] for p in prompts]
        outputs = frontend.run()
        result = [outputs[h.seq_id] for h in handles]
        report = dict(ep=args.ep, dtype=args.dtype, expert_precision=args.expert_precision,
            prompts=prompts, outputs=result,
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            measurement='synchronizing correctness diagnostic; not a performance benchmark')
        (args.output / 'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        frontend.shutdown()


if __name__ == '__main__':
    main()
