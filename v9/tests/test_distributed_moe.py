"""Four-GPU TP2 x EP2 reference test, including an empty EP source rank.

torchrun --standalone --nproc-per-node=4 v9/tests/test_distributed_moe.py
Tiny synthetic weights only; no checkpoint or private data is used.
"""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from engine.process_groups import create_groups
from layers.moe import MoEMLP
from layers.parallel import configure_parallel, tp_allreduce
from types import SimpleNamespace


def main():
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', device_id=torch.device('cuda', torch.cuda.current_device()))
    tp, ep = rank % 2, rank // 2
    groups = create_groups(rank, 2, 1, 2, 1)
    configure_parallel(tp_size=2, tp_rank=tp, ep_size=2, ep_rank=ep,
                       tp_group=groups['tp'], ep_group=groups['ep'])
    cfg = SimpleNamespace(hidden_size=16, num_experts=8, num_experts_per_tok=2,
                          moe_intermediate_size=32, norm_topk_prob=True)
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(42)
        w13 = (torch.randn(8, 64, 16, device='cuda') * .1).to(dtype)
        w2 = (torch.randn(8, 16, 32, device='cuda') * .1).to(dtype)
        model = MoEMLP(cfg).to(device='cuda', dtype=dtype)
        for expert in range(8):
            model.experts.weight_loader(model.experts.w13, w13[expert, :32], (expert, 0))
            model.experts.weight_loader(model.experts.w13, w13[expert, 32:], (expert, 1))
            model.experts.weight_loader(model.experts.w2, w2[expert], (expert, None))
        for sizes in ((0, 7), (3, 5)):
            # Different EP origins have different token counts; TP peers agree.
            torch.manual_seed(100 + ep)
            x = torch.randn(sizes[ep], 16, device='cuda', dtype=dtype)
            ids = ((torch.arange(sizes[ep]*2, device='cuda').view(-1, 2)*3 + ep) % 8).long()
            weights = torch.tensor([.3, .7], device='cuda', dtype=dtype).expand(sizes[ep], 2)
            actual = tp_allreduce(model._ep_forward_alltoallv(x, ids, weights))
            reference = torch.zeros_like(x, dtype=torch.float32)
            # FP32 full-expert oracle, independent of sharding and collectives.
            for token in range(len(x)):
                for route in range(2):
                    e = int(ids[token, route])
                    gate, up = (x[token].float() @ w13[e].float().t()).chunk(2)
                    value = (torch.nn.functional.silu(gate)*up) @ w2[e].float().t()
                    reference[token] += value*weights[token, route].float()
            error = (actual.float()-reference).abs().max().item() if len(x) else 0
            atol, rtol = (2e-6, 2e-5) if dtype == torch.float32 else (2e-3, 3e-2)
            passed = torch.allclose(actual.float(), reference, atol=atol, rtol=rtol)
            ok = torch.tensor(int(passed), device='cuda')
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            assert ok.item(), (rank, str(dtype), sizes, error)
            records.append(dict(rank=rank, dtype=str(dtype), source_sizes=sizes,
                                max_abs_error=error, atol=atol, rtol=rtol, passed=True))
    gathered = [None]*4 if rank == 0 else None
    dist.gather_object(records, gathered, dst=0)
    if rank == 0:
        print(json.dumps({'passed': True, 'scope': 'Synthetic TP2 x EP2 numerical oracle; not full-model HF parity',
                          'records': gathered}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
