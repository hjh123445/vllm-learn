"""Packed EP dispatch must equal the unpacked all-to-allv reference bitwise.

Four Gloo processes (DP2 x TP2 -> EP4) replay unequal and empty per-DP routes
through ``MoEMLP`` twice: once with the reference path and once with
VLLM_LEARN_EP_PACKED=1.  Packing changes the wire layout only; the token
order, expert invocation order and source-side weighting are identical.
TP peers reproduce the same inputs by seeding with the DP index only.
"""
import multiprocessing as mp
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.process_groups import create_groups
from layers.parallel import configure_parallel
from layers.moe import MoEMLP


def worker(rank, dp_size, tp_size, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}",
                            rank=rank, world_size=dp_size * tp_size)
    dp, tp = divmod(rank, tp_size)
    groups = create_groups(rank, tp_size, 1, dp_size, 1, True)
    configure_parallel(tp_size=tp_size, tp_rank=tp, dp_size=dp_size, dp_rank=dp,
                       enable_expert_parallel=True, tp_group=groups["tp"],
                       ep_group=groups["ep"])
    config = SimpleNamespace(hidden_size=16, num_experts=8, num_experts_per_tok=2,
                             moe_intermediate_size=32, norm_topk_prob=True)
    torch.manual_seed(51)
    w13 = torch.randn(8, 64, 16) * .1
    w2 = torch.randn(8, 16, 32) * .1
    router = torch.randn(8, 16) * .1
    results = {}
    with torch.inference_mode():
        for mode in ("reference", "packed"):
            os.environ['VLLM_LEARN_EP_PACKED'] = '1' if mode == 'packed' else '0'
            model = MoEMLP(config)
            model.router.weight.copy_(router)
            for expert in range(8):
                model.experts.weight_loader(model.experts.w13, w13[expert, :32], (expert, 0))
                model.experts.weight_loader(model.experts.w13, w13[expert, 32:], (expert, 1))
                model.experts.weight_loader(model.experts.w2, w2[expert], (expert, None))
            outs = []
            for case in range(4):
                sizes = [0 if (case == 0 and d == 0) or (case == 1 and d != 0)
                         or case == 3 else 1 + d * 3 for d in range(dp_size)]
                tokens = sizes[dp]
                torch.manual_seed(101 + dp)      # TP peers reproduce x
                x = torch.randn(tokens, 16)
                outs.append(model(x))
            results[mode] = outs
        for case, (a, b) in enumerate(zip(results["reference"], results["packed"])):
            assert a.shape == b.shape, (case, a.shape, b.shape)
            assert torch.equal(a, b), f"case {case}: packed dispatch diverged " \
                f"(max abs {(a - b).abs().max().item() if a.numel() else 0.})"
        output.put((rank, "ok"))
    dist.destroy_process_group()


@pytest.mark.parametrize("dp,tp", [(1, 4), (2, 2), (4, 1)])
def test_packed_dispatch_matches_reference(dp, tp, tmp_path):
    context = mp.get_context("spawn")
    output = context.Queue()
    processes = [context.Process(target=worker, args=(rank, dp, tp,
        str(tmp_path / "rendezvous"), output)) for rank in range(4)]
    try:
        for process in processes:
            process.start()
        results = [output.get(timeout=60) for _ in processes]
        assert len({rank for rank, _ in results}) == 4
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
        output.close()
