"""Opt-in NCCL check through production EngineCore/worker_main and loader.

Creates a tiny public-free synthetic checkpoint/tokenizer, then exercises real
initialization, warmup, generation, cancellation and shutdown. Not a benchmark.
"""
import gc
import json
import os
from pathlib import Path
import socket
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.core import EngineCore
from samplingParams import SamplingParams

pytestmark = pytest.mark.skipif(os.getenv("PP_TEST_BACKEND") != "nccl",
                                reason="explicit four-GPU correctness run only")


def checkpoint(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from safetensors.torch import save_file
    path.mkdir()
    vocab = {f"t{i}": i for i in range(64)}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
        WordLevel(vocab, unk_token="t0")), unk_token="t0", eos_token="t63")
    tokenizer.save_pretrained(str(path))
    text = dict(model_type="qwen3_moe", hidden_size=16, head_dim=4,
                num_attention_heads=4, num_key_value_heads=2,
                num_hidden_layers=8, num_experts=4, num_experts_per_tok=2,
                moe_intermediate_size=16, intermediate_size=16,
                norm_topk_prob=True, vocab_size=64, rms_norm_eps=1e-6,
                rope_theta=10000., max_position_embeddings=128)
    # AutoConfig describes the real model family; only text weights are used.
    (path / "config.json").write_text(json.dumps(dict(
        model_type="qwen3_omni_moe", thinker_config=dict(text_config=text))))
    rng = torch.Generator().manual_seed(712)
    weights = {"thinker.model.embed_tokens.weight": torch.randn(64, 16, generator=rng) * .1,
               "thinker.model.norm.weight": torch.ones(16),
               "thinker.lm_head.weight": torch.randn(64, 16, generator=rng) * .1}
    for layer in range(8):
        prefix = f"thinker.model.layers.{layer}."
        for name, shape in dict(
            **{"self_attn.q_proj.weight": (16, 16), "self_attn.k_proj.weight": (8, 16),
               "self_attn.v_proj.weight": (8, 16), "self_attn.o_proj.weight": (16, 16),
               "mlp.gate.weight": (4, 16)}).items():
            weights[prefix + name] = torch.randn(shape, generator=rng) * .1
        for name, shape in (("input_layernorm", (16,)), ("post_attention_layernorm", (16,)),
                            ("self_attn.q_norm", (4,)), ("self_attn.k_norm", (4,))):
            weights[prefix + name + ".weight"] = torch.ones(shape)
        for expert in range(4):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                weights[prefix + f"mlp.experts.{expert}.{projection}.weight"] = (
                    torch.randn(16, 16, generator=rng) * .1)
    save_file(weights, str(path / "model.safetensors"))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_production_worker_initialization_and_cached_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_LEARN_REFERENCE_MATH", "1")
    monkeypatch.setenv("VLLM_LEARN_ATTENTION_BACKEND", "sdpa_reference")
    monkeypatch.setenv("VLLM_LEARN_EXPERT_BACKEND", "loop")
    monkeypatch.setenv("VLLM_LEARN_EP_SHARD_REPLICATED_INPUT", "1")
    monkeypatch.setenv("VLLM_LEARN_EP_LOCAL_ROUTER", "0")
    monkeypatch.setenv("VLLM_LEARN_WORKER_TIMEOUT_S", "90")
    path = tmp_path / "tiny-omni"
    checkpoint(path)
    outputs = []
    prompts = [[1, 2], list(range(2, 12)), list(range(3, 28)), [5, 6, 7], [1, 2]]
    for pp, tp, ep in ((1, 1, False), (4, 1, False), (2, 2, True), (2, 2, False)):
        core = EngineCore(str(path), pp_size=pp, tp_size=tp, enable_expert_parallel=ep,
            master_port=free_port(), gpus=list(range(pp * tp)),
            pipeline_microbatch_tokens=8, config_kwargs=dict(
                max_model_len=128, max_num_batched_tokens=8, max_num_seqs=16,
                gpu_memory_utilization=.98, max_kvcache_blocks=16,
                enforce_eager=True, batch_execution_mode="per_request"))
        try:
            result = core.generate(prompts, SamplingParams(temperature=0., max_tokens=4, ignore_eos=True))
            outputs.append([tokens for _, tokens in result])
            if pp > 1:
                abandoned = core.add_request([1, 2], SamplingParams(max_tokens=4))
                core.step()
                assert abandoned.seq_id in core.scheduler.in_flight
                core.abort_request(abandoned.seq_id)
                assert abandoned.block_table
                while not core.is_finished():
                    core.step()
                assert not abandoned.block_table and not abandoned.completion_token_ids
        finally:
            core.shutdown()
        assert all(p.exitcode == 0 for p in core.procs)
        del core
        gc.collect()
        torch.cuda.empty_cache()
    assert all(result == outputs[0] for result in outputs[1:]), outputs


@pytest.mark.parametrize("dp,tp,pp", [(2, 2, 1), (2, 1, 2), (4, 1, 1)])
def test_production_dp_ep_unequal_batches_and_idle_replicas(tmp_path, monkeypatch, dp, tp, pp):
    from engine.frontend import DPFrontend
    monkeypatch.setenv("VLLM_LEARN_REFERENCE_MATH", "1")
    monkeypatch.setenv("VLLM_LEARN_ATTENTION_BACKEND", "sdpa_reference")
    monkeypatch.setenv("VLLM_LEARN_EXPERT_BACKEND", "loop")
    monkeypatch.setenv("VLLM_LEARN_WORKER_TIMEOUT_S", "90")
    path = tmp_path / "tiny-omni"
    checkpoint(path)
    config = dict(max_model_len=128, max_num_batched_tokens=8, max_num_seqs=16,
                  gpu_memory_utilization=.98, max_kvcache_blocks=16, enforce_eager=True)
    prompts = [[1, 2], list(range(2, 12)), list(range(3, 28)), [5, 6, 7], [1, 2]]
    sampling = SamplingParams(temperature=0., max_tokens=4, ignore_eos=True)
    serial = EngineCore(str(path), config_kwargs=config)
    try:
        reference = [tokens for _, tokens in serial.generate(prompts, sampling)]
    finally:
        serial.shutdown()
    del serial
    gc.collect()
    torch.cuda.empty_cache()
    # Match attention TP and per-DP scheduler/GEMM shapes in the oracle.
    # Serial BF16 greedy parity is retained as a diagnostic, not assumed:
    # TP changes attention reduction shapes even with EP disabled.
    baseline = DPFrontend(str(path), dp_size=dp, tp_size=tp, pp_size=pp,
        master_port=free_port(), gpus=list(range(dp * tp * pp)),
        pipeline_microbatch_tokens=8, config_kwargs=config)
    try:
        baseline_handles = [baseline.add_request(prompt, sampling)[0] for prompt in prompts]
        baseline_output = baseline.run()
        matched_reference = [baseline_output[h.seq_id] for h in baseline_handles]
        single_handle, _ = baseline.add_request(prompts[0], sampling, dp_rank=dp - 1)
        matched_single = baseline.run()[single_handle.seq_id]
    finally:
        baseline.shutdown()
    frontend = DPFrontend(str(path), dp_size=dp, tp_size=tp, pp_size=pp,
        enable_expert_parallel=True, master_port=free_port(),
        gpus=list(range(dp * tp * pp)), pipeline_microbatch_tokens=8,
        config_kwargs=config)
    processes = list(frontend.cores)
    try:
        handles = [frontend.add_request(prompt, sampling)[0] for prompt in prompts]
        outputs = frontend.run()
        actual = [outputs[h.seq_id] for h in handles]
        print(dict(scope="BF16 production, matched DP/TP/PP baseline with EP disabled",
                   serial=reference, matched_tp=matched_reference, coupled_ep=actual), flush=True)
        assert actual == matched_reference
        # Only the final DP replica receives work; every other replica must
        # still join every expert dispatch/combine, across multiple PP ticks.
        handle, _ = frontend.add_request(prompts[0], sampling, dp_rank=dp - 1)
        assert frontend.run() == {handle.seq_id: matched_single}
    finally:
        frontend.shutdown()
    frontend.shutdown()  # idempotent, no extra collective on already closed workers
    assert all(process.exitcode == 0 for process in processes)


def test_dp2tp2_ep4_fixed_prefix_logits_against_expert_tp(tmp_path, monkeypatch):
    """Same inputs/prefixes; compare logits before free-generation drift.

    Whole-expert BF16 and expert-TP BF16 have different rounding boundaries.
    This tolerance oracle complements, and does NOT replace, strict greedy
    parity above (whose failure must remain visible in the test results).
    """
    from engine.frontend import DPFrontend
    from utils.observed_runner import ObservedModelRunner
    monkeypatch.setenv("VLLM_LEARN_REFERENCE_MATH", "1")
    monkeypatch.setenv("VLLM_LEARN_ATTENTION_BACKEND", "sdpa_reference")
    monkeypatch.setenv("VLLM_LEARN_EXPERT_BACKEND", "loop")
    monkeypatch.setenv("VLLM_LEARN_WORKER_TIMEOUT_S", "90")
    path = tmp_path / "tiny-omni"
    checkpoint(path)
    prompts = [[1, 2], list(range(2, 12)), list(range(3, 28)), [5, 6, 7], [1, 2]]
    tables = []
    for enable_ep in (False, True):
        capture = tmp_path / ("ep" if enable_ep else "expert_tp")
        monkeypatch.setenv("VLLM_LEARN_LOGITS_DIR", str(capture))
        frontend = DPFrontend(str(path), dp_size=2, tp_size=2,
            runner_cls=ObservedModelRunner,
            enable_expert_parallel=enable_ep, master_port=free_port(), gpus=list(range(4)),
            config_kwargs=dict(max_model_len=128, max_num_batched_tokens=8,
                max_num_seqs=16, gpu_memory_utilization=.98, max_kvcache_blocks=16,
                enforce_eager=True))
        try:
            for forced_prefix in ([], [54], [54, 60]):
                for prompt in prompts:
                    frontend.add_request(prompt + forced_prefix,
                        SamplingParams(temperature=0., max_tokens=1, ignore_eos=True))
                assert len(frontend.run()) == len(prompts)
            # A one-sided request tests empty DP execution as part of a model.
            frontend.add_request([1, 2], SamplingParams(temperature=0., max_tokens=1,
                                 ignore_eos=True), dp_rank=1)
            assert len(frontend.run()) == 1
        finally:
            frontend.shutdown()
        table = {}
        for file in sorted(capture.rglob("*.pt")):
            record = torch.load(file, weights_only=True)
            for row, seq_id in enumerate(record["seq_ids"]):
                table[record["dp_rank"], seq_id] = record["logits"][row]
        assert len(table) == 16
        tables.append(table)
    assert tables[0].keys() == tables[1].keys()
    errors, greedy_mismatches = [], 0
    for key in tables[0]:
        baseline, actual = tables[0][key], tables[1][key]
        torch.testing.assert_close(actual, baseline, atol=.02, rtol=.03)
        errors.append(float((actual - baseline).abs().max()))
        greedy_mismatches += int(actual.argmax() != baseline.argmax())
    print(dict(scope="16 forced-prefix synthetic BF16 logits vectors, DP2TP2 EP4 vs expert TP",
               max_abs=max(errors), greedy_mismatches=greedy_mismatches,
               atol=.02, rtol=.03, task_accuracy=False), flush=True)
