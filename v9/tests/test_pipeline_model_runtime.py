"""CPU/Gloo integration using the real Qwen model, runner, scheduler and IPC.

Tiny synthetic FP32 weights, reference attention/math, no model download and
no GPU. Exercises PP4, PP2xEP2 and PP2xTP2 against a serial model oracle.
Requires a Linux PyTorch environment with Gloo and the project's dependencies.
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
from engine.core import EngineCore, rebuild_specs
from engine.executor import MultiprocessExecutor
from engine.mq import make_channels
from engine.process_groups import create_groups, create_pipeline_edge_groups
from engine.scheduler import Scheduler
from engine.worker_health import wait_for_workers
from layers.parallel import STATE, configure_parallel
from samplingParams import SamplingParams


def model_config():
    return SimpleNamespace(hidden_size=16, head_dim=4, num_attention_heads=4,
                           num_key_value_heads=2, num_hidden_layers=8,
                           num_experts=4, num_experts_per_tok=2,
                           moe_intermediate_size=16, norm_topk_prob=True,
                           vocab_size=64, rms_norm_eps=1e-6, rope_theta=10000.)


def engine_config():
    return SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=32,
                           eos=-1, kvcache_block_size=256, num_kvcache_blocks=16,
                           enable_prefix_caching=True, batch_execution_mode="batched",
                           hf_config=model_config())


def make_runner(weights):
    from engine.model_runner import ModelRunner
    from engine.sampler import Sampler
    from layers.moe import GroupedExperts
    from models.qwen3_moe import Qwen3MoeForCausalLM
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = engine_config()
    runner.device = "cuda" if os.getenv("PP_TEST_BACKEND") == "nccl" else "cpu"
    runner.dtype, runner.block_size = torch.float32, 256
    runner.model = Qwen3MoeForCausalLM(runner.config.hf_config, max_position=128)
    runner.model.to(runner.device)
    runner.sampler = Sampler()
    with torch.no_grad():
        for name, module in runner.model.named_modules():
            for field, param in module.named_parameters(recurse=False):
                loaded = weights[f"{name}.{field}"]
                if isinstance(module, GroupedExperts):
                    for expert in range(module.e_start, module.e_start + module.num_experts):
                        if field == "w13":
                            gate, up = loaded[expert].chunk(2, dim=0)
                            module.weight_loader(param, gate, (expert, 0))
                            module.weight_loader(param, up, (expert, 1))
                        else:
                            module.weight_loader(param, loaded[expert], (expert, None))
                elif hasattr(param, "weight_loader"):
                    param.weight_loader(param, loaded)
                else:
                    param.copy_(loaded)
    layers = runner.model.model.pp_end - runner.model.model.pp_start
    kv_heads = runner._per_rank_kv_heads()
    runner.kv_cache = torch.zeros(2, layers, 16, 256, kv_heads, 4, device=runner.device)
    runner._assign_layer_caches(runner.kv_cache)
    runner.model.eval()
    return runner


def real_model_worker(rank, pp, tp, ep, init_file, commands, results, weights):
    torch.set_num_threads(1)
    inner, size = tp, pp * tp
    backend = os.getenv("PP_TEST_BACKEND", "gloo")
    kwargs = {}
    if backend == "nccl":
        torch.cuda.set_device(rank)
        kwargs["device_id"] = torch.device("cuda", rank)
    dist.init_process_group(backend, init_method=f"file://{init_file}",
                            rank=rank, world_size=size, **kwargs)
    groups = create_groups(rank, tp, 1, 1, pp, ep)
    incoming, outgoing = create_pipeline_edge_groups(rank, inner, pp)
    configure_parallel(tp_size=tp, tp_rank=rank % tp,
                       enable_expert_parallel=ep,
                       pp_size=pp, pp_rank=rank // inner,
                       tp_group=groups["tp"], ep_group=groups["ep"])
    from engine.pipeline_transport import PipelineTransport
    STATE.pipeline_transport = PipelineTransport(pp, STATE.pp_rank, inner,
        device="cuda" if backend == "nccl" else "cpu",
        lanes=1, credit=2, recv_group=incoming, send_group=outgoing)
    runner = make_runner(weights)
    results.send({"type": "ready"})
    response_pending = False
    while True:
        msg = commands.recv()
        if msg["type"] == "shutdown":
            break
        specs = rebuild_specs(msg)
        tokens = runner.run(specs, msg["is_prefill"], microbatch_id=msg["microbatch_id"])
        if rank == (pp - 1) * inner:
            if response_pending:
                results.wait_ack()
            results.send(dict(type="output", tokens=tokens,
                              seq_ids=[s.seq_id for s in specs],
                              microbatch_id=msg["microbatch_id"]), wait_ack=False)
            response_pending = True
    if response_pending:
        results.wait_ack()
    STATE.pipeline_transport.shutdown()
    dist.destroy_process_group()


def run_requests(core):
    prompts = [[1, 2], list(range(2, 12)), list(range(3, 28)), [5, 6, 7], [1, 2]]
    seqs = [core.add_request(p, SamplingParams(temperature=0., max_tokens=4)) for p in prompts]
    for _ in range(200):
        if core.is_finished():
            break
        core.step()
    assert core.is_finished()
    assert not core.scheduler.block_manager.used_block_ids
    return [list(s.completion_token_ids) for s in seqs]


def build_core(pp, executor):
    core = EngineCore.__new__(EngineCore)
    core._closed = False
    core.pp_size, core.max_concurrent_batches = pp, pp
    core.pipeline_microbatch_tokens, core.pipeline_mode = 8, "optimized"
    core._worker_timeout = 30
    core.config = engine_config()
    core.scheduler = Scheduler(core.config)
    core.executor = executor
    core.procs, core.broadcast_q, core.response_q = [], None, None
    core._initialize_batch_queue()
    return core


@pytest.mark.parametrize("pp,tp,ep", [(4, 1, False), (2, 2, True), (2, 2, False)])
def test_pipeline_model_matches_serial_cached_generation(tmp_path, monkeypatch, pp, tp, ep):
    monkeypatch.setenv("VLLM_LEARN_REFERENCE_MATH", "1")
    monkeypatch.setenv("VLLM_LEARN_ATTENTION_BACKEND", "sdpa_reference")
    monkeypatch.setenv("VLLM_LEARN_EXPERT_BACKEND", "loop")
    monkeypatch.setenv("VLLM_LEARN_EP_SHARD_REPLICATED_INPUT", "1")
    monkeypatch.setenv("VLLM_LEARN_EP_LOCAL_ROUTER", "0")
    torch.set_num_threads(1)
    configure_parallel()
    STATE.pipeline_transport = None
    from models.qwen3_moe import Qwen3MoeForCausalLM
    torch.manual_seed(712)
    model = Qwen3MoeForCausalLM(model_config(), max_position=128)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.ones_like(param) if "norm" in name else torch.randn_like(param) * .1)
    weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    serial = build_core(1, None)
    serial.inproc_runner = make_runner(weights)
    reference = run_requests(serial)
    size = pp * tp
    broadcast, responses = make_channels(size)
    context = mp.get_context("spawn")
    processes = [context.Process(target=real_model_worker, args=(
        rank, pp, tp, ep, str(tmp_path / "rendezvous"), broadcast[rank][1],
        responses[rank][1], weights)) for rank in range(size)]
    core = None
    try:
        for process in processes:
            process.start()
        for pair in responses:
            wait_for_workers(pair[0].notify, processes, 30)
            assert pair[0].recv()["type"] == "ready"
        executor = MultiprocessExecutor([p[0] for p in broadcast],
            responses[(pp - 1) * tp][0], processes, timeout=30, max_pending=pp)
        core = build_core(pp, executor)
        core.procs = processes
        core.broadcast_q, core.response_q = [p[0] for p in broadcast], [p[0] for p in responses]
        assert run_requests(core) == reference
        core.shutdown()
        assert all(p.exitcode == 0 for p in processes)
    finally:
        if core is not None and not core._closed:
            core._abort_workers()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
        for pair in broadcast + responses:
            for endpoint in pair:
                endpoint.close()
        configure_parallel()
        STATE.pipeline_transport = None
