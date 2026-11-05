"""Real spawn + SHM IPC regression for >2 in-flight pipeline batches.

The stage arithmetic is synthetic CPU work. This verifies RPC flow/ACK order,
not NCCL performance or numerical equivalence of the language model.
"""
import multiprocessing as mp
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.executor import MultiprocessExecutor
from engine.mq import make_channels


def stage_worker(rank, size, commands, results, links, accepted, gate, malformed=False):
    # The last stage intentionally delays reading control messages. Earlier
    # stage senders must still progress; a broadcast-wide ACK barrier cannot.
    if rank == size - 1:
        if not gate.wait(10):
            raise TimeoutError("test gate was not released")
    response_pending = False
    while True:
        message = commands.recv()
        if message["type"] == "shutdown":
            break
        batch_id = message["microbatch_id"]
        accepted.put((rank, batch_id))
        if rank:
            incoming_id, tokens = links[rank - 1].get(timeout=10)
            assert incoming_id == batch_id
        else:
            tokens = [s["last_token"] for s in message["scheduled"]]
        tokens = [token + rank + 1 for token in tokens]
        if rank < size - 1:
            links[rank].put((batch_id, tokens))
        else:
            if response_pending:
                results.wait_ack()
            results.send(dict(type="output", microbatch_id=batch_id + int(malformed),
                              seq_ids=[s["seq_id"] for s in message["scheduled"]],
                              tokens=tokens), wait_ack=False)
            response_pending = True
        if batch_id == 7 and rank > 0:
            # Rank 0 can exit while other ranks have not ACKed shutdown yet.
            time.sleep(.3)
    if response_pending:
        results.wait_ack()
    commands.close()
    results.close()


class Pipeline:
    def __init__(self, size=4, malformed=False):
        self.size = size
        self.context = mp.get_context("spawn")
        self.broadcast, self.responses = make_channels(size)
        self.links = [self.context.Queue() for _ in range(size - 1)]
        self.accepted = self.context.Queue()
        self.gate = self.context.Event()
        self.processes = [self.context.Process(target=stage_worker, args=(
            rank, size, self.broadcast[rank][1], self.responses[rank][1],
            self.links, self.accepted, self.gate, malformed)) for rank in range(size)]
        for process in self.processes:
            process.start()
        self.executor = MultiprocessExecutor(
            [pair[0] for pair in self.broadcast], self.responses[-1][0],
            self.processes, timeout=8, max_pending=8)

    def close(self):
        self.gate.set()
        self.executor.abort()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
        for pair in self.broadcast + self.responses:
            for endpoint in pair:
                endpoint.close()
        for q in (*self.links, self.accepted):
            q.close()
            q.join_thread()


def message(index):
    return dict(type="schedule", microbatch_id=index,
                scheduled=[dict(seq_id=index, last_token=index)])


def test_per_stage_submission_and_output_drain_break_both_ack_barriers():
    pipeline = Pipeline()
    try:
        futures = [pipeline.executor.execute_model(message(i)) for i in range(8)]
        observed = set()
        deadline = time.monotonic() + 8
        while (0, 3) not in observed and time.monotonic() < deadline:
            observed.add(pipeline.accepted.get(timeout=8))
        assert (0, 3) in observed  # last stage has not even ACKed batch 0
        assert not any(f.done() for f in futures)
        with pytest.raises(RuntimeError, match="queue is full"):
            pipeline.executor.execute_model(message(8))
        pipeline.gate.set()
        # Do not consume results until all 8 model RPCs have been enqueued.
        # The receiver must ACK the single output slot independently of Core.
        assert [f.result(timeout=8)["tokens"] for f in futures] == [[i + 10] for i in range(8)]
        pipeline.executor.shutdown()
        for process in pipeline.processes:
            process.join(timeout=5)
            assert process.exitcode == 0
    finally:
        pipeline.close()


@pytest.mark.parametrize("failure", ["wrong_batch", "worker_exit"])
def test_protocol_or_worker_failure_reaches_every_outstanding_future(failure):
    pipeline = Pipeline(size=3, malformed=failure == "wrong_batch")
    try:
        futures = [pipeline.executor.execute_model(message(i)) for i in range(4)]
        if failure == "worker_exit":
            pipeline.processes[-1].terminate()
            pipeline.processes[-1].join(timeout=3)
        pipeline.gate.set()
        for future in futures:
            with pytest.raises(RuntimeError, match="protocol mismatch|worker exited"):
                future.result(timeout=8)
    finally:
        pipeline.close()
