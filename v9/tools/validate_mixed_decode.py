"""Verify that admitting a prompt preserves live decode logits and KV exactly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(rank, args):
    import torch
    import torch.distributed as dist
    from engine.process_groups import create_groups
    from layers.parallel import configure_parallel
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{args.port}',
        rank=rank, world_size=8, device_id=torch.device(f'cuda:{rank}'))
    groups = create_groups(rank, tp_size=8, enable_expert_parallel=True)
    for group in groups.values():
        if group is not None:
            dist.all_reduce(torch.zeros(1, device='cuda'), group=group)
    configure_parallel(tp_size=8, tp_rank=rank, tp_group=groups['tp'],
        ep_group=groups['ep'], enable_expert_parallel=True)
    from config import Config
    from engine.model_runner import ModelRunner
    from engine.sequence import Sequence
    from engine.core import _spec_msg, rebuild_specs
    from samplingParams import SamplingParams
    runner = ModelRunner(Config(args.model, max_model_len=1024, max_num_seqs=8,
        max_num_batched_tokens=2048, max_kvcache_blocks=32,
        gpu_memory_utilization=.75))
    generator = torch.Generator().manual_seed(20261108)
    def forward(seqs, prefill):
        # Exercise the production IPC snapshot, not only direct Sequence calls.
        return runner.run(rebuild_specs(_spec_msg(seqs, prefill)), prefill)
    rows = []
    try:
        seqs = []
        for i, length in enumerate((17, 255, 511)):
            seq = Sequence(torch.randint(100, 30000, (length,), generator=generator).tolist(),
                           SamplingParams(temperature=0.))
            seq.block_table = list(range(i*4, i*4+seq.num_blocks))
            seq.num_scheduled_tokens = length
            seqs.append(seq)
        forward(seqs, True)
        for step in range(4):
            for seq in seqs:
                seq.num_cached_tokens = len(seq)
                seq.append_token(3 + step)
                seq.num_scheduled_tokens = 1
                if len(seq.block_table) < seq.num_blocks:
                    seq.block_table.append(seq.block_table[0]+len(seq.block_table))
            new = Sequence(torch.randint(100, 30000, (23+step,), generator=generator).tolist(),
                           SamplingParams(temperature=0.))
            new.block_table = [12]
            new.num_scheduled_tokens = len(new)
            captured = []
            handle = runner.sampler.register_forward_pre_hook(
                lambda _module, inputs: captured.append(inputs[0].detach().clone()))
            try:
                with torch.inference_mode():
                    before = runner.kv_cache.clone()
                    expected_tokens = forward(seqs, False)
                    expected = captured.pop()
                    expected_kv = runner.kv_cache[:, :, :12].clone()
                    runner.kv_cache.copy_(before)
                    actual_tokens = forward([seqs[0], new, seqs[1], seqs[2]], True)
                    assert len(captured) == 2, 'mixed batch did not split into decode/prefill'
                    actual = captured[0]
                    row = dict(step=step, exact_logits=torch.equal(actual, expected),
                        max_abs=(actual.float()-expected.float()).abs().max().item(),
                        exact_live_kv=torch.equal(expected_kv, runner.kv_cache[:, :, :12]),
                        token_order_equal=[actual_tokens[i] for i in (0,2,3)] == expected_tokens)
                    rows.append(row)
                    if not all(row[k] for k in ('exact_logits', 'exact_live_kv', 'token_order_equal')):
                        raise AssertionError(row)
            finally:
                handle.remove()
        root = Path(__file__).resolve().parents[1]
        result = dict(rank=rank, graph=args.graph, rows=rows,
            replays=getattr(runner, 'graph_replay_count', 0), source_sha256={str(p.relative_to(root)):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')})
        (Path(args.output)/f'rank{rank}.json').write_text(json.dumps(result, indent=2))
        print('MIXED_DECODE_PASS', rank, args.graph, rows, flush=True)
    finally:
        runner.shutdown()
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--port', type=int, default=31941)
    ap.add_argument('--graph', action='store_true')
    args = ap.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    os.environ.update(VLLM_LEARN_MM='0', VLLM_LEARN_BATCH_INVARIANT='1',
        VLLM_LEARN_REFERENCE_QK='0', VLLM_LEARN_ATTENTION_BACKEND='triton',
        VLLM_LEARN_EP_DENSE_DECODE='0', VLLM_LEARN_EP_SPARSE_DECODE='1', VLLM_LEARN_DENSE_EXPERT_BACKEND='bmm',
        VLLM_LEARN_DECODE_GRAPH='1' if args.graph else '0')
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=8, join=True)


if __name__ == '__main__':
    main()
