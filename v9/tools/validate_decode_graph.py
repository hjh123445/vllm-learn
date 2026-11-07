"""Explicit multi-GPU graph/eager correctness probe; never a throughput run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(rank, args):
    import faulthandler
    faulthandler.dump_traceback_later(args.traceback_timeout, repeat=False)
    import torch
    import torch.distributed as dist
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{args.port}',
                            rank=rank, world_size=args.tp, device_id=torch.device(f'cuda:{rank}'))
    from engine.process_groups import create_groups
    from layers.parallel import configure_parallel
    groups = create_groups(rank, tp_size=args.tp, enable_expert_parallel=True)
    for group in groups.values():
        if group is not None:
            dist.all_reduce(torch.zeros(1, device='cuda'), group=group)
    configure_parallel(tp_size=args.tp, tp_rank=rank, tp_group=groups['tp'],
                       ep_group=groups['ep'], enable_expert_parallel=True)
    from config import Config
    from engine.model_runner import ModelRunner
    from engine.sequence import Sequence
    from samplingParams import SamplingParams
    stride = (args.max_model_len + 255) // 256
    config = Config(args.model, max_model_len=args.max_model_len,
                    max_num_batched_tokens=args.prefill_tokens,
                    max_num_seqs=args.max_batch,
                    max_kvcache_blocks=args.max_batch * stride,
                    decode_graph_reserve_bytes=int(args.graph_reserve_gib * 2**30),
                    gpu_memory_utilization=.8)
    class CheckedRunner(ModelRunner):
        def _capture_decode_graph(self, batch, blocks):
            print(f'rank{rank}: capture begin B={batch}', flush=True)
            super()._capture_decode_graph(batch, blocks)
            print(f'rank{rank}: capture end B={batch}', flush=True)
    runner = CheckedRunner(config)
    print(f'rank{rank}: runner ready', flush=True)
    rows = []
    generator = torch.Generator().manual_seed(args.seed)
    try:
        expected_buckets = sorted({b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
                                   if b <= args.max_batch} | {args.max_batch})
        assert sorted(runner.decode_graphs) == expected_buckets
        assert config.num_kvcache_blocks >= args.max_batch * stride
        # Padded/exact buckets exercise batch shrink and
        # crossing physical block boundary 255 -> 256. Fixed prefixes prevent
        # generated-token divergence from changing the comparison inputs.
        for case_index, n in enumerate(args.batches):
            length = args.lengths[case_index % len(args.lengths)]
            seqs = []
            for i in range(n):
                row_length = args.lengths[i % len(args.lengths)] if args.mixed_lengths else length
                tokens = (torch.randint(100, 30000, (row_length,), generator=generator).tolist()
                          if args.random_inputs else [i + 1] * row_length)
                seq = Sequence(tokens, SamplingParams(temperature=0.))
                seq.block_table = [i * stride + j for j in range(seq.num_blocks)]
                seq.num_scheduled_tokens = row_length
                seqs.append(seq)
            print(f'rank{rank}: prefill B={n} L={length}', flush=True)
            # Prefill small groups so a 256-request correctness probe does
            # not build one enormous sparse expert activation. Every rank
            # uses the same deterministic groups and collective order.
            chunk, chunk_tokens = [], 0
            for seq in seqs:
                if chunk and chunk_tokens + len(seq) > args.prefill_tokens:
                    runner.run(chunk, True)
                    chunk, chunk_tokens = [], 0
                chunk.append(seq)
                chunk_tokens += len(seq)
            if chunk:
                runner.run(chunk, True)
            for step in range(args.steps):
                for seq in seqs:
                    seq.append_token(3 if step == 0 else generated[seq.seq_id]); seq.num_scheduled_tokens = 1
                    if len(seq.block_table) < seq.num_blocks:
                        seq.block_table.append(seq.block_table[0] + len(seq.block_table))
                with torch.inference_mode():
                    kv_before = runner.kv_cache.clone()
                    ids, pos = runner.prepare_decode(seqs)
                    eager = runner.model.compute_logits(runner.model(ids, pos)).clone()
                    kv_eager = runner.kv_cache.clone()
                    runner.kv_cache.copy_(kv_before)
                    state = runner.decode_graphs[runner._pick_decode_graph(n)]
                    graph = runner._replay_decode_graph(state, seqs).clone()
                    kv_graph_equal = torch.equal(kv_eager, runner.kv_cache)
                    runner.kv_cache.copy_(kv_before)
                    from utils.context import set_context
                    set_context(False, slot_mapping=state['slots'], context_lens=state['lens'], block_tables=state['bt'])
                    padded = runner.model.compute_logits(runner.model(state['ids'], state['pos']))[:n].clone()
                    kv_padded_equal = torch.equal(kv_eager, runner.kv_cache)
                    permutation_equal = True
                    if args.permute:
                        runner.kv_cache.copy_(kv_before)
                        reversed_logits = runner._replay_decode_graph(state, list(reversed(seqs))).clone()
                        permutation_equal = torch.equal(reversed_logits, graph.flip(0))
                    runner.kv_cache.copy_(kv_eager)
                torch.cuda.synchronize()
                print(f'rank{rank}: compared B={n}', flush=True)
                diff = (padded.float() - graph.float()).abs()
                equal = torch.equal(padded, graph)
                rank0_logits = graph.clone()
                dist.broadcast(rank0_logits, src=0, group=groups['tp'])
                ranks_equal = torch.equal(rank0_logits, graph)
                rows.append(dict(batch=n, prefix_length=length, context_lengths=[len(s) for s in seqs], decode_step=step, bucket=state['ids'].shape[0],
                                 exact_logits=equal, max_abs=diff.max().item(),
                                 argmax_equal=torch.equal(padded.argmax(-1), graph.argmax(-1)),
                                 tp_ranks_equal=ranks_equal,
                                 kv_graph_equal=kv_graph_equal, kv_padded_equal=kv_padded_equal,
                                 permutation_equal=permutation_equal,
                                 unpadded_max_abs=(eager.float()-graph.float()).abs().max().item(),
                                 unpadded_argmax_equal=torch.equal(eager.argmax(-1), graph.argmax(-1))))
                (Path(args.output) / f'progress_rank{rank}.json').write_text(json.dumps(rows, indent=2))
                if not (equal and ranks_equal and kv_graph_equal and kv_padded_equal and permutation_equal):
                    raise AssertionError(rows[-1])
                # Batch invariance is a separate, stronger property than graph
                # equivalence. Preserve its measurements even when it fails.
                if args.strict_batch_invariant and rows[-1]['unpadded_max_abs'] != 0:
                    raise AssertionError(rows[-1])
                generated = {seq.seq_id: token for seq, token in zip(seqs, eager.argmax(-1).tolist())}
                if args.permute: seqs.reverse()
        # Verify capture itself never changes any live KV values.
        print(f'rank{rank}: sentinel capture begin', flush=True)
        with torch.inference_mode():
            runner.kv_cache[:, :, :config.num_kvcache_blocks].fill_(.125)
            sentinel_bucket = min(4, args.max_batch)
            runner.decode_graphs[sentinel_bucket]['graph'].reset()
            runner._capture_decode_graph(sentinel_bucket, stride)
        unchanged = bool((runner.kv_cache[:, :, :config.num_kvcache_blocks] == .125).all().item())
        assert unchanged
        print(f'rank{rank}: sentinel capture passed', flush=True)
        profile_file = None
        if args.profile and rank == 0:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                runner._replay_decode_graph(runner.decode_graphs[runner._pick_decode_graph(len(seqs))], seqs)
                torch.cuda.synchronize()
            profile_file = str(Path(args.output) / 'graph_rank0.json')
            prof.export_chrome_trace(profile_file)
        # All ranks must participate in NCCL graph replay, even if only rank0
        # records CUPTI. The no-profile peers execute the same replay.
        elif args.profile:
            runner._replay_decode_graph(runner.decode_graphs[runner._pick_decode_graph(len(seqs))], seqs)
            torch.cuda.synchronize()
        result = dict(rank=rank, model=args.model, torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), rows=rows, live_kv_unchanged=unchanged,
                      buckets=sorted(runner.decode_graphs), replays=runner.graph_replay_count,
                      profile=profile_file, seed=args.seed, random_inputs=args.random_inputs, steps=args.steps, permute=args.permute, mixed_lengths=args.mixed_lengths,
                      batch_invariant=os.getenv('VLLM_LEARN_BATCH_INVARIANT', '0') == '1',
                      strict_batch_invariant=args.strict_batch_invariant,
                      graph_capture_bytes=runner._graph_capture_bytes,
                      graph_capture_count=runner.graph_capture_count,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      max_batch=args.max_batch, lengths=args.lengths,
                      source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):
                          hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).resolve().parents[1].rglob('*.py')
                          if '__pycache__' not in p.parts},
                      measurement='correctness only; same-kernel eager, not HF parity')
        (Path(args.output) / f'rank{rank}.json').write_text(json.dumps(result, indent=2))
        faulthandler.cancel_dump_traceback_later()
    finally:
        state = None
        print(f'rank{rank}: runner shutdown', flush=True)
        runner.shutdown()
        import gc
        gc.collect()
        print(f'rank{rank}: distributed shutdown', flush=True)
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--tp', type=int, default=8)
    ap.add_argument('--port', type=int, default=31917)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--strict-batch-invariant', action='store_true',
                    help='also require unpadded eager to be bitwise equal; BF16 may fail')
    ap.add_argument('--batch-invariant', action='store_true')
    ap.add_argument('--random-inputs', action='store_true',
                    help='use reproducible non-repeating token prefixes instead of repeated IDs')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--permute', action='store_true')
    ap.add_argument('--mixed-lengths', action='store_true')
    ap.add_argument('--max-batch', type=int, default=8)
    ap.add_argument('--batches', type=int, nargs='+',
                    help='ordered batch sizes; default descends from max-batch to 1')
    ap.add_argument('--lengths', type=int, nargs='+', default=[17, 255, 256, 511])
    ap.add_argument('--max-model-len', type=int, default=1024)
    ap.add_argument('--prefill-tokens', type=int, default=4096)
    ap.add_argument('--graph-reserve-gib', type=float, default=1.)
    ap.add_argument('--traceback-timeout', type=int, default=600)
    args = ap.parse_args()
    if args.steps < 1: ap.error('--steps must be positive')
    if args.max_batch < 1 or args.max_model_len < 1: ap.error('capacities must be positive')
    if args.batches is None: args.batches = list(range(args.max_batch, 0, -1))
    if not args.batches or min(args.batches) < 1 or max(args.batches) > args.max_batch:
        ap.error('batches must be within [1, max-batch]')
    if min(args.lengths) < 1 or max(args.lengths) + args.steps > args.max_model_len:
        ap.error('prefix length plus decode steps must fit max-model-len')
    if args.prefill_tokens < max(args.lengths): ap.error('prefill-tokens must fit longest prefix')
    if args.graph_reserve_gib <= 0: ap.error('graph-reserve-gib must be positive')
    if args.batch_invariant: os.environ['VLLM_LEARN_BATCH_INVARIANT'] = '1'
    Path(args.output).mkdir(parents=True, exist_ok=True)
    os.environ.update(VLLM_LEARN_DECODE_GRAPH='1', VLLM_LEARN_EP_DENSE_DECODE='1',
                      VLLM_LEARN_REFERENCE_QK='0', VLLM_LEARN_ATTENTION_BACKEND='triton',
                      VLLM_LEARN_EXPERT_BACKEND='torch_grouped', VLLM_LEARN_MM='0')
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=args.tp, join=True)


if __name__ == '__main__':
    main()
