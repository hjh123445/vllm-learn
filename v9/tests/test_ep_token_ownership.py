"""Four-GPU numerical and route-count oracle for replicated-input ownership.

torchrun --standalone --nproc-per-node=4 v9/tests/test_ep_token_ownership.py
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from engine.process_groups import create_groups
from layers.moe import MoEMLP
from layers.parallel import configure_parallel


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
    with torch.inference_mode():
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(42)
            w13 = (torch.randn(8, 64, 16, device='cuda')*.1).to(dtype)
            w2 = (torch.randn(8, 16, 32, device='cuda')*.1).to(dtype)
            router = (torch.randn(8, 16, device='cuda')*.1).to(dtype)
            model = MoEMLP(cfg).to(device='cuda', dtype=dtype)
            model.router.weight.copy_(router)
            for e in range(8):
                model.experts.weight_loader(model.experts.w13, w13[e,:32], (e,0))
                model.experts.weight_loader(model.experts.w13, w13[e,32:], (e,1))
                model.experts.weight_loader(model.experts.w2, w2[e], (e,None))
            for tokens in (0, 1, 3, 8, 17):
                torch.manual_seed(100+tokens)
                x = torch.randn(tokens,16,device='cuda',dtype=dtype)
                model.shard_replicated_input = False
                baseline = model(x)
                model.shard_replicated_input = True
                model.experts._profile_enabled = True
                model.experts._profile_token_counts = None
                actual = model(x)
                atol, rtol = (2e-6,2e-5) if dtype==torch.float32 else (2e-3,3e-2)
                passed = torch.allclose(actual.float(),baseline.float(),atol=atol,rtol=rtol)
                local = model.experts._profile_token_counts
                count = torch.tensor(0 if local is None else int(local.sum()),device='cuda')
                dist.all_reduce(count,group=groups['ep'])
                assert count.item() == tokens*2, (rank,tokens,count.item())
                ok = torch.tensor(int(passed),device='cuda')
                dist.all_reduce(ok,op=dist.ReduceOp.MIN)
                error = (actual-baseline).abs().max().item() if tokens else 0
                assert ok.item(), (rank,str(dtype),tokens,error)
                records.append(dict(dtype=str(dtype),tokens=tokens,max_abs_error=error,
                    total_expert_routes_per_tp_group=int(count),expected_routes=tokens*2,passed=True))
                model.experts._profile_enabled = False
    if rank==0:
        print(json.dumps(dict(passed=True,scope='TP2 EP2 replicated-input ownership oracle',records=records)))
    dist.destroy_process_group()


if __name__=='__main__':
    main()
