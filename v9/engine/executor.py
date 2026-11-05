"""Non-blocking, ordered worker RPC over the existing shared-memory channels.

Only this class touches worker IPC after initialization. Independent per-rank
senders avoid making stage 0 wait for stage P-1 to accept the previous batch.
A separate receiver drains output ACKs. Scheduler state stays on the Core thread.
"""

from collections import OrderedDict
from concurrent.futures import Future
import queue
import threading

from engine.mq import SHUTDOWN
from engine.worker_health import wait_for_workers


class MultiprocessExecutor:
    def __init__(self, broadcast_channels, response_channel, processes,
                 timeout, max_pending):
        self._response = response_channel
        self._processes = processes
        self._timeout = timeout
        self._max_pending = max_pending
        self._commands = [queue.Queue() for _ in broadcast_channels]
        self._pending = OrderedDict()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._error = None
        self._closed = False
        self._senders = [
            threading.Thread(target=self._send_loop, args=(channel, commands, rank),
                             name=f"pp-submit-{rank}", daemon=True)
            for rank, (channel, commands) in enumerate(zip(broadcast_channels, self._commands))
        ]
        self._threads = [*self._senders,
            threading.Thread(target=self._receive_loop, name="pp-results", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def execute_model(self, message):
        """Submit a detached input snapshot; return before worker ACK/compute."""
        future = Future()
        # GPU work cannot be cancelled by cancelling a Python Future. Request
        # cancellation is settled by the scheduler after this result arrives.
        future.set_running_or_notify_cancel()
        batch_id = message["microbatch_id"]
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Worker executor failed") from self._error
            if self._closed:
                raise RuntimeError("Worker executor is closed")
            if len(self._pending) >= self._max_pending:
                raise RuntimeError("Worker executor queue is full")
            if batch_id in self._pending:
                raise ValueError(f"duplicate batch id {batch_id}")
            ids = tuple(spec["seq_id"] for spec in message["scheduled"])
            self._pending[batch_id] = (future, ids)
            for commands in self._commands:
                commands.put(message)
            self._condition.notify_all()
        return future

    def _wait(self, connection):
        wait_for_workers(connection, self._processes, self._timeout,
                         stop_event=self._stop)

    def _fail(self, error):
        with self._condition:
            if self._error is None:
                self._error = error
            pending = list(self._pending.values())
            self._pending.clear()
            self._stop.set()
            self._condition.notify_all()
        for future, _ in pending:
            if not future.done():
                future.set_exception(error)

    def _send_loop(self, channel, commands, rank):
        try:
            while not self._stop.is_set():
                try:
                    message = commands.get(timeout=.1)
                except queue.Empty:
                    continue
                channel.send(message, wait_ack=False)
                if message["type"] == "shutdown":
                    # Earlier stages may exit normally before later stages
                    # ACK shutdown. Only the target worker must remain alive.
                    channel.wait_ack(wait=lambda connection: wait_for_workers(
                        connection, [self._processes[rank]], self._timeout,
                        stop_event=self._stop))
                    return
                channel.wait_ack(wait=self._wait)
        except Exception as error:
            self._fail(error)

    def _receive_loop(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._stop.is_set())
                    if self._stop.is_set():
                        return
                    batch_id, (future, ids) = next(iter(self._pending.items()))
                self._wait(self._response.notify)
                output = self._response.recv()  # ACK immediately; don't await Core.
                if (output.get("type") != "output"
                        or output.get("microbatch_id") != batch_id
                        or tuple(output.get("seq_ids", ())) != ids
                        or len(output.get("tokens", ())) != len(ids)):
                    raise RuntimeError(f"worker output protocol mismatch for batch {batch_id}")
                with self._condition:
                    if self._stop.is_set():
                        return
                    self._pending.pop(batch_id)
                    future.set_result(output)
        except Exception as error:
            self._fail(error)

    def abort(self):
        """Wake IPC waits before the owner kills workers/closes shared slots."""
        self._fail(RuntimeError("Worker executor aborted"))
        self._join_threads(2)

    def _join_threads(self, timeout):
        for thread in self._threads:
            thread.join(timeout=timeout)
        if any(thread.is_alive() for thread in self._threads):
            raise TimeoutError("Worker IPC threads have not stopped")

    def shutdown(self):
        """Called only after EngineCore has settled every submitted batch."""
        with self._condition:
            if self._pending:
                raise RuntimeError("Drain model results before executor shutdown")
            if self._error is not None:
                raise RuntimeError("Worker executor failed") from self._error
            self._closed = True
            for commands in self._commands:
                commands.put(SHUTDOWN)
        for sender in self._senders:
            sender.join(timeout=self._timeout)
        if any(sender.is_alive() for sender in self._senders):
            raise TimeoutError("Worker shutdown broadcast timed out")
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._join_threads(2)
        if self._error is not None:
            raise RuntimeError("Worker executor failed during shutdown") from self._error
