"""Distributed exact eager/piecewise Graph check through ModelRunner + PP P2P.

Tiny synthetic checkpoint, real NCCL transports. Each topology is compared
against its OWN eager arithmetic, never a claim of cross-topology BF16 parity.
"""
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
    from engine.process_groups import ParallelTopology, create_groups, create_pipeline_edge_groups
    from layers.parallel import configure_parallel, STATE
    from layers.dcp import set_dcp
    from engine.model_runner import ModelRunner
    from engine.pipeline_transport import PipelineTransport
    from engine.sequence import Sequence
    from samplingParams import SamplingParams
    from config import Config
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    topo = ParallelTopology(args.tp, args.dp, args.pp, args.cp, args.ep)
    d, p, t, c = topo.coords(rank)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{args.port}',
                            rank=rank, world_size=topo.world_size,
                            device_id=torch.device('cuda', rank))
    groups = create_groups(rank, args.tp, args.cp, args.dp, args.pp, args.ep)
    incoming, outgoing = None, None
    if args.pp > 1:
        incoming, outgoing = create_pipeline_edge_groups(rank, args.tp * args.cp, args.pp, args.dp)
    for group in [*groups.values(), incoming, outgoing]:
        if group is not None:
            dist.all_reduce(torch.zeros(1, device='cuda'), group=group)
    configure_parallel(tp_size=args.tp, tp_rank=t, tp_group=groups['tp'],
        ep_group=groups['ep'], cp_size=args.cp, cp_rank=c, pp_size=args.pp, pp_rank=p,
        dp_size=args.dp, dp_rank=d, enable_expert_parallel=args.ep)
    set_dcp(active=args.cp > 1, size=args.cp, rank=c, group=groups['cp'])
    if args.pp > 1:
        STATE.pipeline_transport = PipelineTransport(args.pp, p, args.tp * args.cp,
            recv_group=incoming, send_group=outgoing, rank_offset=d * topo.local_world_size)
    runner = None
    rows = []
    try:
        runner = ModelRunner(Config(args.model, max_model_len=1024, max_num_seqs=8,
            max_num_batched_tokens=1024, max_kvcache_blocks=32,
            decode_graph_reserve_bytes=512 * 2**20, gpu_memory_utilization=.8))
        assert runner.graph_execution_mode == 'piecewise'
        # Compare the entire pool, including unused slots. Initialize those
        # slots so allocator leftovers/NaN != NaN cannot masquerade as writes.
        runner.kv_cache.zero_()
        # One round uneven DP loads, another only last DP has requests.
        for round_id in range(2):
            n = (3 if d == 0 else 1) if round_id == 0 else (2 if d == args.dp - 1 else 0)
            seqs = []
            for i in range(n):
                length = (17, 255, 256)[i % 3]
                tokens = [(j * 7 + d * 5 + i + 1) % 60 + 1 for j in range(length)]
                seq = Sequence(tokens, SamplingParams(temperature=0., max_tokens=8, ignore_eos=True))
                seq.block_table = [i * 4 + j for j in range(seq.num_blocks)]
                seq.num_scheduled_tokens = length
                seqs.append(seq)
            runner.run(seqs, True, microbatch_id=round_id * 100)
            for step in range(3):
                for seq in seqs:
                    seq.num_cached_tokens = len(seq)
                    seq.append_token(5 + step)
                    seq.num_scheduled_tokens = 1
                    if len(seq.block_table) < seq.num_blocks:
                        seq.block_table.append(seq.block_table[0] + len(seq.block_table))
                seen = []
                hook = runner.sampler.register_forward_pre_hook(lambda _, values: seen.append(values[0].clone()))
                before = runner.kv_cache.clone()
                manager = runner._piecewise_graphs
                try:
                    runner._piecewise_graphs = None
                    runner._decode_graph_requested = False
                    eager_tokens = runner.run(seqs, False, microbatch_id=round_id * 100 + step * 2 + 1)
                    eager_logits = seen.pop() if seen else None
                    expected_kv = runner.kv_cache.clone()
                    runner.kv_cache.copy_(before)
                    runner._piecewise_graphs = manager
                    runner._decode_graph_requested = True
                    graph_tokens = runner.run(seqs, False, microbatch_id=round_id * 100 + step * 2 + 2)
                    graph_logits = seen.pop() if seen else None
                    exact = (eager_logits is None and graph_logits is None) or (
                        eager_logits is not None and graph_logits is not None and torch.equal(eager_logits, graph_logits))
                    row = dict(round=round_id, step=step, local_requests=n, logits_exact=exact,
                        tokens_exact=eager_tokens == graph_tokens, kv_exact=torch.equal(expected_kv, runner.kv_cache),
                        kv_mismatch_count=int((expected_kv != runner.kv_cache).sum().item()),
                        segment_replays=manager.replays, segment_captures=manager.captures)
                    rows.append(row)
                    assert row['logits_exact'] and row['tokens_exact'] and row['kv_exact'], row
                finally:
                    runner._piecewise_graphs = manager
                    runner._decode_graph_requested = True
                    hook.remove()
            if n:
                assert manager.replays > 0, 'no computation segment replayed'
        out = dict(rank=rank, topology=vars(args), rows=rows,
                   source_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):
                       hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in Path(__file__).resolve().parents[1].rglob('*.py')},
                   graph_mode=runner.graph_execution_mode, captures=manager.captures, replays=manager.replays)
        (Path(args.output) / f'rank{rank}.json').write_text(json.dumps(out, indent=2))
        print('PARALLEL_GRAPH_PASS', rank, args.tp, args.dp, args.pp, args.cp, args.ep, flush=True)
    finally:
        if runner is not None:
            runner.shutdown()
        if STATE.pipeline_transport is not None:
            STATE.pipeline_transport.shutdown()
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    for name in ('tp', 'dp', 'pp', 'cp'):
        ap.add_argument('--' + name, type=int, default=1)
    ap.add_argument('--ep', action='store_true')
    ap.add_argument('--port', type=int, default=31983)
    args = ap.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    os.environ.update(VLLM_LEARN_DECODE_GRAPH='1', VLLM_LEARN_DECODE_GRAPH_MODE='piecewise',
        VLLM_LEARN_EP_SPARSE_DECODE='0', VLLM_LEARN_EP_DENSE_DECODE='0',
        VLLM_LEARN_BATCH_INVARIANT='1', VLLM_LEARN_REFERENCE_QK='0',
        VLLM_LEARN_ATTENTION_BACKEND='triton', VLLM_LEARN_MM='0', OMP_NUM_THREADS='2')
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=args.tp * args.dp * args.pp * args.cp, join=True)


if __name__ == '__main__':
    main()
