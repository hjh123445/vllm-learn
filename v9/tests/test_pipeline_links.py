"""Multi-stage transport oracle: variable shapes, lanes, credits and groups.

Run with torchrun --standalone --nproc-per-node=4 ... --backend gloo|nccl.
No checkpoint or private data is needed.
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from engine.pipeline_transport import PipelineTransport
from engine.process_groups import create_pipeline_edge_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['gloo', 'nccl'], default='gloo')
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.backend == 'nccl':
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', torch.cuda.current_device()) if args.backend == 'nccl' else torch.device('cpu')
    dist.init_process_group(args.backend)
    rank, size = dist.get_rank(), dist.get_world_size()
    if size < 3:
        raise ValueError('Use at least three stages to cover both incoming and outgoing groups')
    incoming, outgoing = create_pipeline_edge_groups(rank, 1, size)
    for group in (incoming, outgoing):
        if group is not None:
            dist.all_reduce(torch.zeros(1, device=device), group=group)
    try:
        for dedicated in (False, True):
            for asynchronous in (False, True):
                pipe = PipelineTransport(size, rank, 1, lanes=2, credit=1,
                    async_enabled=asynchronous, device=str(device),
                    recv_group=incoming if dedicated else None,
                    send_group=outgoing if dedicated else None)
                for mb, tokens in enumerate((0, 1, 7, 2, 13, 1, 5)):
                    lane = mb % 2
                    if rank == 0:
                        hidden = torch.full((tokens, 5), mb, dtype=torch.bfloat16, device=device)
                        residual = torch.full_like(hidden, -mb)
                    else:
                        hidden, residual, actual_mb, actual_lane = pipe.recv_forward(lane)
                        assert (actual_mb, actual_lane) == (mb, lane)
                        assert hidden.shape == (tokens, 5)
                        expected = mb + sum(range(1, rank))
                        torch.testing.assert_close(hidden, torch.full_like(hidden, expected), rtol=0, atol=0)
                        torch.testing.assert_close(residual, torch.full_like(residual, -mb), rtol=0, atol=0)
                        hidden = hidden + rank
                    pipe.send_forward(hidden, residual, lane, mb)
                pipe.shutdown()
                dist.barrier()
                if rank == 0:
                    print(f'PASS dedicated={dedicated} async={asynchronous} stages={size}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
