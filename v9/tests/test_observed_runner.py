"""CPU-only checks for opt-in capture and production/diagnostic separation."""
import json
import multiprocessing as mp
from datetime import timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.core import EngineCore
from engine.model_runner import ModelRunner
from engine.sampler import Sampler
from engine.sequence import Sequence
from layers.parallel import configure_parallel, STATE
from utils.observed_runner import ObservedModelRunner, ObservedPipelineTransport


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block()])
        self.pp_start, self.pp_end = 0, 1

    def forward(self, x):
        return self.layers[0](x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body()

    def forward(self, x):
        return self.model(x)


def seq():
    value = Sequence([1, 2])
    value.num_scheduled_tokens = 2
    return value


@pytest.fixture
def cpu_runner(monkeypatch):
    configure_parallel()
    STATE.pipeline_transport = None
    for name in ("VLLM_LEARN_LOGITS_DIR", "VLLM_LEARN_ACTIVATIONS_DIR", "VLLM_LEARN_TRACE_DIR",
                 "VLLM_LEARN_CUDA_PROFILE", "VLLM_LEARN_METRICS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def init(self, config, dtype=None):
        self.config = config
        self.model = TinyModel()
        self.sampler = Sampler()
        self.kv_cache = torch.zeros(2)
        self.run([seq()], True)  # Memory sizing warmup must not be captured.

    def run(self, seqs, is_prefill, microbatch_id=0):
        if getattr(self.config, "per_request", False) and len(seqs) > 1:
            return sum((self.run([s], is_prefill, microbatch_id=microbatch_id) for s in seqs), [])
        logits = self.model(torch.ones(len(seqs), 4))
        return self.sampler(logits, torch.zeros(len(seqs))).tolist()

    monkeypatch.setattr(ModelRunner, "__init__", init)
    monkeypatch.setattr(ModelRunner, "run", run)


def test_engine_factory_and_normal_runner_ignore_capture_environment(cpu_runner, monkeypatch, tmp_path):
    config = SimpleNamespace(num_kvcache_blocks=16, kvcache_block_size=256,
        max_num_seqs=4, max_num_batched_tokens=16, enable_prefix_caching=False)
    monkeypatch.setattr("config.Config", lambda *_, **__: config)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                        lambda *_, **__: SimpleNamespace(eos_token_id=3))
    capture = tmp_path / "logits"
    monkeypatch.setenv("VLLM_LEARN_LOGITS_DIR", str(capture))
    normal = EngineCore("synthetic")
    try:
        assert type(normal.inproc_runner) is ModelRunner
        normal.inproc_runner.run([seq()], True)
        assert not capture.exists()
    finally:
        normal.shutdown()
    observed = EngineCore("synthetic", runner_cls=ObservedModelRunner)
    try:
        assert type(observed.inproc_runner) is ObservedModelRunner
        assert not capture.exists()  # Warmup excluded.
        observed.inproc_runner.run([seq()], True)
        assert len(list(capture.glob("*.pt"))) == 1
    finally:
        observed.shutdown()
    assert not observed.inproc_runner.sampler._forward_pre_hooks


def test_window_exports_cpu_trace_and_removes_hooks(cpu_runner, monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_LEARN_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_LEARN_CUDA_PROFILE", "1")
    monkeypatch.setenv("VLLM_LEARN_PROFILE_MODE", "window")
    monkeypatch.setenv("VLLM_LEARN_PROFILE_START_MESSAGE", "2")
    monkeypatch.setenv("VLLM_LEARN_PROFILE_MESSAGES", "1")
    runner = ObservedModelRunner(SimpleNamespace())
    try:
        for batch in (11, 12, 13):
            runner.run([seq()], True, microbatch_id=batch)
        assert all(not m._forward_hooks for m in runner.model.modules())
    finally:
        runner.shutdown()
    runner.shutdown()  # No duplicate stop/export.
    trace = json.loads((tmp_path / "cuda_rank0.json").read_text())
    names = {e["name"] for e in trace["traceEvents"]}
    assert "stage0.layer0.attention" in names and "aten::linear" in names
    assert "rank0.stage0.message2.mb12.prefill" in names
    assert "rank0.stage0.message3.mb13.prefill" not in names
    metadata = json.loads((tmp_path / "capture_rank0.json").read_text())
    assert metadata["cuda_activities"] is False
    assert [m["microbatch_id"] for m in metadata["captured_messages"]] == [12]
    timeline = json.loads((tmp_path / "pipeline_rank0.json").read_text())
    assert len(timeline["traceEvents"]) == 3
    assert timeline["metadata"]["measurement"] == "CPU calls, not GPU duration"


def test_logits_per_request_and_independent_dp_namespace(cpu_runner, monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_LEARN_LOGITS_DIR", str(tmp_path))
    STATE.replica_size, STATE.replica_rank = 2, 1
    runner = ObservedModelRunner(SimpleNamespace(per_request=True))
    try:
        requests = [seq(), seq()]
        runner.run(requests, True)
        files = sorted((tmp_path / "dp1").glob("*.pt"))
        assert len(files) == 2 and not list(tmp_path.glob("*.pt"))
        records = [torch.load(f, weights_only=True) for f in files]
        assert [r["seq_ids"] for r in records] == [[s.seq_id] for s in requests]
        assert all(r["logits"].shape == (1, 4) and r["dp_rank"] == 1 for r in records)
    finally:
        runner.shutdown()
        configure_parallel()


def test_engine_and_model_sources_have_no_capture_dependencies():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("torch.profiler", "autograd.profiler", "record_function", "nvtx",
        "VLLM_LEARN_TRACE", "VLLM_LEARN_PROFILE", "VLLM_LEARN_CUDA_PROFILE",
        "VLLM_LEARN_LOGITS", "VLLM_LEARN_ACTIVATIONS", "VLLM_LEARN_METRICS",
        "module_profiling", "activation_capture", "observed_runner")
    for directory in ("engine", "models", "layers"):
        for path in (root / directory).rglob("*.py"):
            source = path.read_text(encoding="utf-8-sig")
            assert not any(word in source for word in forbidden), str(path)


def _transport_peer(rank, init_file, directory, asynchronous, output):
    import torch.distributed as dist
    import os
    torch.set_num_threads(1)
    os.environ["VLLM_LEARN_TRACE_DIR"] = directory
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    transport = ObservedPipelineTransport(2, rank, 1, device="cpu",
                                         credit=1, async_enabled=asynchronous)
    try:
        for batch in range(6):
            hidden = torch.full((batch + 1, 4), float(batch))
            residual = hidden + 1
            if rank == 0:
                transport.send_forward(hidden, residual, microbatch_id=batch)
            else:
                h, r = transport.recv_forward(microbatch_id=batch,
                    shape=hidden.shape, dtype=torch.float32)
                torch.testing.assert_close(h, hidden)
                torch.testing.assert_close(r, residual)
        transport.shutdown()
        output.put((rank, transport.events))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_observed_transport_preserves_real_gloo_payloads(tmp_path, asynchronous):
    ctx = mp.get_context("spawn")
    output = ctx.Queue()
    processes = [ctx.Process(target=_transport_peer,
        args=(rank, str(tmp_path / "rendezvous"), str(tmp_path), asynchronous, output))
        for rank in range(2)]
    try:
        for process in processes:
            process.start()
        records = [output.get(timeout=40) for _ in processes]
        for rank, events in records:
            assert len(events) == 6
            assert {e["name"] for e in events} == {"pp.send" if rank == 0 else "pp.recv"}
            assert [e["args"]["microbatch_id"] for e in events] == list(range(6))
            assert all(e["dur"] >= 0 for e in events)
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
        output.close()
