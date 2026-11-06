"""Real all-to-allv oracle with unequal DP batches and empty source replicas.

Four Gloo processes by default; PP_TEST_BACKEND=nccl uses four CUDA devices.
Only tiny synthetic weights. No full checkpoint, dataset or speed benchmark.
"""
import multiprocessing as mp
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.process_groups import create_groups
from layers.parallel import configure_parallel
from layers.moe import MoEMLP


def worker(rank, dp_size, tp_size, rendezvous, output):
    torch.set_num_threads(1)
    backend = os.getenv("PP_TEST_BACKEND", "gloo")
    device = "cpu"
    kwargs = {}
    if backend == "nccl":
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        kwargs["device_id"] = torch.device(device)
    dist.init_process_group(backend, init_method=f"file://{rendezvous}" if rendezvous else "env://", rank=rank,
                            world_size=dp_size * tp_size, **kwargs)
    dp, tp = divmod(rank, tp_size)
    groups = create_groups(rank, tp_size, 1, dp_size, 1, True)
    configure_parallel(tp_size=tp_size, tp_rank=tp, dp_size=dp_size, dp_rank=dp,
                       enable_expert_parallel=True, tp_group=groups["tp"], ep_group=groups["ep"])
    config = SimpleNamespace(hidden_size=16, num_experts=8, num_experts_per_tok=2,
                             moe_intermediate_size=32, norm_topk_prob=True)
    records = []
    with torch.inference_mode():
        for dtype in ((torch.float32, torch.bfloat16) if backend == "nccl" else (torch.float32,)):
            torch.manual_seed(51)
            w13 = (torch.randn(8, 64, 16, device=device) * .1).to(dtype)
            w2 = (torch.randn(8, 16, 32, device=device) * .1).to(dtype)
            router = (torch.randn(8, 16, device=device) * .1).to(dtype)
            model = MoEMLP(config).to(device=device, dtype=dtype)
            assert model.experts.inter == 32 and model.experts.num_experts == 8 // (dp_size * tp_size)
            model.router.weight.copy_(router)
            for expert in range(8):
                model.experts.weight_loader(model.experts.w13, w13[expert, :32], (expert, 0))
                model.experts.weight_loader(model.experts.w13, w13[expert, 32:], (expert, 1))
                model.experts.weight_loader(model.experts.w2, w2[expert], (expert, None))
            for case in range(4):
                # Per-DP sources differ; TP peers agree, even when some own no rows.
                sizes = [0 if (case == 0 and d == 0) or (case == 1 and d != 0) or case == 3
                         else 1 + d * 3 for d in range(dp_size)]
                tokens = sizes[dp]
                torch.manual_seed(101 + dp)
                x = torch.randn(tokens, 16, device=device, dtype=dtype)
                # Count evaluated routes at the test boundary; production
                # expert code does not carry experiment-only counters.
                with patch.object(model.experts, "compute_sorted",
                                  wraps=model.experts.compute_sorted) as compute:
                    actual = model(x)
                route_count = sum(call.args[0].shape[0] for call in compute.call_args_list)
                ids, weights = model.router(x)
                reference = torch.zeros_like(x, dtype=torch.float32)
                for token in range(tokens):
                    for k in range(2):
                        expert = int(ids[token, k])
                        gate, up = (x[token].float() @ w13[expert].float().t()).chunk(2)
                        value = (torch.nn.functional.silu(gate) * up) @ w2[expert].float().t()
                        reference[token] += value * weights[token, k].float()
                torch.testing.assert_close(actual.float(), reference,
                    atol=2e-6 if dtype == torch.float32 else 2e-3,
                    rtol=2e-5 if dtype == torch.float32 else 3e-2)
                count = torch.tensor(route_count, device=device)
                dist.all_reduce(count, group=groups["ep"])
                assert count.item() == sum(sizes) * 2  # no TP duplicate routes, no dropped DP requests
                records.append((str(dtype), sizes, count.item()))
    output.put((rank, records))
    dist.destroy_process_group()


@pytest.mark.parametrize("dp,tp", [(1, 4), (2, 2), (4, 1)])
@pytest.mark.parametrize("precision", ['model', 'float32'])
def test_coupled_ep_matches_full_expert_oracle(tmp_path, monkeypatch, dp, tp, precision):
    monkeypatch.setenv('VLLM_LEARN_EXPERT_PRECISION', precision)
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
