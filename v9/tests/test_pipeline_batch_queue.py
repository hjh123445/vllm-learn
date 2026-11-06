"""Real scheduler/KV ownership with controlled asynchronous worker Futures.

No model, torch or GPU required. These tests exercise EngineCore's runtime
queue, not the idealized pipeline planner.
"""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.core import EngineCore, rebuild_specs
from engine.scheduler import Scheduler
from engine.sequence import Sequence
from samplingParams import SamplingParams


def configuration(blocks=64, prefix=False):
    return SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=512, eos=-1,
                           kvcache_block_size=256, num_kvcache_blocks=blocks,
                           enable_prefix_caching=prefix)


class ControlledExecutor:
    def __init__(self, immediate=True):
        self.immediate = immediate
        self.messages = []
        self.futures = []
        self.closed = False

    def execute_model(self, message):
        future = Future()
        self.messages.append(message)
        self.futures.append(future)
        if self.immediate:
            self.finish(len(self.messages) - 1)
        return future

    def finish(self, index, **overrides):
        message = self.messages[index]
        output = dict(type="output", microbatch_id=message["microbatch_id"],
                      seq_ids=[s["seq_id"] for s in message["scheduled"]],
                      tokens=[(s["last_token"] + s["seq_len"]) % 97 + 1
                              for s in message["scheduled"]])
        output.update(overrides)
        self.futures[index].set_result(output)

    def shutdown(self):
        self.closed = True

    def abort(self):
        self.closed = True


def core_for_test(pp=4, depth=None, executor=None, config=None):
    core = EngineCore.__new__(EngineCore)
    core._closed = False
    core.pp_size = pp
    core.max_concurrent_batches = pp if depth is None else depth
    core.pipeline_microbatch_tokens = 64
    core._worker_timeout = 2
    core.config = config or configuration()
    core.scheduler = Scheduler(core.config)
    core.executor = executor or ControlledExecutor()
    core.procs, core.broadcast_q, core.response_q = [], [], []
    core._initialize_batch_queue()
    return core


def add_requests(core, prompts, max_tokens=4):
    return [core.add_request(p, SamplingParams(temperature=0., max_tokens=max_tokens))
            for p in prompts]


def run_to_end(core):
    for _ in range(2000):
        if core.is_finished():
            return
        core.step()
    raise AssertionError("scheduler did not converge")


def test_results_retire_individually_and_refill_before_tail_finishes():
    executor = ControlledExecutor(immediate=False)
    core = core_for_test(executor=executor)
    seqs = add_requests(core, [[i + 1] for i in range(8)])
    for _ in range(3):
        assert core.step() == []
    assert len(core.batch_queue) == 3
    assert len(core.scheduler.in_flight) == 6
    assert all(not s.completion_token_ids for s in seqs)
    executor.finish(0)
    core.step()  # submit batch 3, retire only batch 0
    assert len(core.batch_queue) == 3
    assert [len(s.completion_token_ids) for s in seqs] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert not executor.futures[3].done()
    executor.finish(1)
    core.step()  # batch 4 reuses requests 0/1 while batch 3 is still outstanding
    assert [s["seq_id"] for s in executor.messages[4]["scheduled"]] == [s.seq_id for s in seqs[:2]]
    assert executor.messages[4]["scheduled"][0]["num_cached_tokens"] == 1
    assert not executor.futures[3].done()
    # The first message is a detached snapshot, not the now-advanced Sequence.
    assert executor.messages[0]["scheduled"][0]["seq_len"] == 1


@pytest.mark.parametrize("prefix", [False, True])
def test_mixed_chunks_and_decode_equal_serial_and_release_all_kv(prefix):
    prompts = [[1], [2] * 13, [3] * 257, [4] * 513, [3] * 257]
    outputs = []
    for pp in (1, 4):
        core = core_for_test(pp=pp, config=configuration(prefix=prefix))
        seqs = add_requests(core, prompts, max_tokens=7)
        run_to_end(core)
        outputs.append([list(s.completion_token_ids) for s in seqs])
        assert all(len(s.completion_token_ids) == 7 for s in seqs)
        assert not core.scheduler.in_flight
        assert not core.scheduler.block_manager.used_block_ids
        assert len(core.scheduler.block_manager.free_block_ids) == 64
        for message in core.executor.messages:
            assert sum(s["num_scheduled_tokens"] for s in message["scheduled"]) <= (64 if pp > 1 else 512)
    assert outputs[0] == outputs[1]


def test_cancelled_inflight_request_keeps_kv_until_its_own_result():
    executor = ControlledExecutor(immediate=False)
    core = core_for_test(executor=executor, config=configuration(blocks=1))
    first = add_requests(core, [[1] * 256])[0]
    core.step()
    blocks = list(first.block_table)
    assert core.abort_request(first.seq_id)
    assert first.block_table == blocks
    assert not first.is_finished
    second = add_requests(core, [[2]])[0]
    assert core.scheduler.schedule() == ([], False)
    assert second.block_table == []
    executor.finish(0)
    core.step()
    assert first.is_finished and not first.block_table
    assert not first.completion_token_ids
    assert core.last_step_events == []
    scheduled, _ = core.scheduler.schedule()
    assert scheduled == [second]
    assert second.block_table == blocks


def test_inflight_requests_cannot_be_rescheduled_or_preempted():
    scheduler = Scheduler(configuration(blocks=3))
    seq = Sequence([1] * 256, SamplingParams(max_tokens=4))
    scheduler.add(seq)
    first, _ = scheduler.schedule(token_budget=32)
    snapshot = (seq.num_cached_tokens, seq.num_scheduled_tokens, list(seq.block_table))
    assert first == [seq]
    assert scheduler.schedule(token_budget=32) == ([], False)
    assert snapshot == (seq.num_cached_tokens, seq.num_scheduled_tokens, seq.block_table)
    with pytest.raises(RuntimeError, match="outstanding GPU"):
        scheduler.preempt(seq)
    scheduler.postprocess(first, [99], True)
    assert seq.num_cached_tokens == 32
    assert not seq.completion_token_ids  # incomplete prefill's sample is ignored
    assert scheduler.schedule(token_budget=32)[0] == [seq]


def test_pending_prefix_is_not_published_for_other_requests():
    scheduler = Scheduler(configuration(prefix=True))
    a, b = [Sequence([5] * 512, SamplingParams(max_tokens=2)) for _ in range(2)]
    scheduler.add(a)
    scheduler.add(b)
    scheduler.schedule(max_seqs=1)
    assert not scheduler.block_manager.hash_to_block_id
    scheduler.schedule(max_seqs=1)
    assert set(a.block_table).isdisjoint(b.block_table)
    scheduler.postprocess([a], [7], True)
    assert scheduler.block_manager.hash_to_block_id


def test_shutdown_drains_pending_results_without_scheduling_more():
    core = core_for_test()
    seqs = add_requests(core, [[1], [2], [3], [4]])
    core.step()
    core.step()
    submitted = len(core.executor.messages)
    core.shutdown()
    assert core._closed and core.executor.closed
    assert len(core.executor.messages) == submitted
    assert not core.batch_queue and not core.scheduler.in_flight
    assert all(s.is_finished for s in seqs)
    assert not core.scheduler.block_manager.used_block_ids


@pytest.mark.parametrize("wrong", [{"microbatch_id": 99}, {"seq_ids": []}, {"tokens": []}])
def test_wrong_result_never_updates_request_or_frees_live_kv(wrong):
    executor = ControlledExecutor(immediate=False)
    core = core_for_test(executor=executor)
    seq = add_requests(core, [[1]])[0]
    core.step()
    executor.finish(0, **wrong)
    with pytest.raises(RuntimeError, match="does not match batch"):
        core.step()
    assert core._closed and executor.closed
    assert not seq.completion_token_ids


def test_worker_failure_closes_core_and_propagates_original_error():
    executor = ControlledExecutor(immediate=False)
    core = core_for_test(executor=executor)
    add_requests(core, [[1]])
    core.step()
    executor.futures[0].set_exception(RuntimeError("injected model failure"))
    with pytest.raises(RuntimeError, match="injected model failure"):
        core.step()
    assert core._closed


def test_empty_schedule_does_not_send_an_empty_gpu_message():
    core = core_for_test()
    assert core.step() == []
    assert not core.last_step_had_work
    assert not core.executor.messages


def test_nondefault_kv_block_size_survives_worker_snapshot():
    config = configuration()
    config.kvcache_block_size = 512
    core = core_for_test(config=config)
    seq = add_requests(core, [[1] * 513])[0]
    core.step()
    spec = rebuild_specs(core.executor.messages[0])[0]
    assert seq.num_blocks == 2
    assert spec.block_size == 512
    assert spec.last_block_num_tokens == 1


def test_schedule_snapshot_preserves_identity_without_lane():
    from engine.core import _spec_msg
    seq = Sequence([3, 7], SamplingParams(temperature=0.))
    seq.num_scheduled_tokens = 2
    seq.block_table = [4]
    message = _spec_msg([seq], True, microbatch_id=5)
    assert message["microbatch_id"] == 5
    assert "pipeline_lane" not in message
    assert rebuild_specs(message)[0][:2] == [3, 7]
    seq.block_table.append(9)
    assert message["scheduled"][0]["block_table"] == [4]
