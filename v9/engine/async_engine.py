"""Online adapter for the v9 EngineCore.

The HTTP layer owns request queues; EngineCore remains the single scheduler
owner and is advanced by one background thread.  Keeping this adapter on the
new EngineCore path makes the server exercise the same worker/MQ/PP code as
offline inference instead of silently falling back to the legacy LLMEngine.
"""

import queue
import threading
import itertools
import time


class EngineOverloaded(RuntimeError):
    """Bounded admission rejected a request before allocating KV state."""


class _StreamState:
    """A request-local token stream (engine thread writes, client reads)."""

    __slots__ = ("q", "sent", "finished", "finish_reason", "error", "times", "capacity")

    def __init__(self, capacity=256):
        self.capacity = capacity
        self.q = queue.Queue(maxsize=capacity + 1)  # reserve the terminal marker
        self.sent = 0
        self.finished = False
        self.finish_reason = None
        self.error = None
        self.times = {"submitted": time.perf_counter()}

    def close(self, reason, error=None):
        if self.finished:
            return
        self.error, self.finish_reason = error, reason
        self.finished = True
        self.times["finished"] = time.perf_counter()
        self.q.put_nowait(None)


class AsyncEngine:
    """Background-thread facade over :class:`engine.core.EngineCore`.

    ``tensor_parallel_size`` sets attention TP; ``enable_expert_parallel``
    shards whole experts over DP x TP. Independent expert sizes are rejected.
    """

    def __init__(self, model: str, **kwargs):
        from transformers import AutoTokenizer
        from engine.core import EngineCore
        self.max_pending_requests = int(kwargs.pop("max_pending_requests", 1024))
        self.stream_buffer_tokens = int(kwargs.pop("stream_buffer_tokens", 256))
        if min(self.max_pending_requests, self.stream_buffer_tokens) < 1:
            raise ValueError("admission and stream buffer limits must be positive")

        tp_size = kwargs.pop("tp_size", kwargs.pop("tensor_parallel_size", 1))
        enable_ep = kwargs.pop("enable_expert_parallel", False)
        pp_size = kwargs.pop("pp_size", kwargs.pop("pipeline_parallel_size", 1))
        dcp_size = kwargs.pop("dcp_size", 1)
        dp_size = kwargs.pop("data_parallel_size", 1)
        gpus = kwargs.pop("gpus", None)
        master_port = kwargs.pop("master_port", 29600)
        pipeline_credit = kwargs.pop("pipeline_credit", 4)
        pipeline_mode = kwargs.pop("pipeline_mode", "optimized")
        pipeline_microbatch_tokens = kwargs.pop("pipeline_microbatch_tokens", 2048)
        max_concurrent_batches = kwargs.pop("max_concurrent_batches", None)
        runner_cls = kwargs.pop("runner_cls", None)
        transport_cls = kwargs.pop("transport_cls", None)
        config_fields = {
            "max_num_batched_tokens", "max_num_seqs", "max_model_len",
            "gpu_memory_utilization", "kvcache_block_size",
            "max_kvcache_blocks", "enable_prefix_caching", "batch_execution_mode",
        }
        unknown = set(kwargs) - config_fields
        if unknown:
            raise TypeError(f"Unknown engine options: {sorted(unknown)}")
        config_kwargs = {k: v for k, v in kwargs.items()
                         if k in config_fields}
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if dp_size > 1:
            raise ValueError(
                "AsyncEngine serves one EngineCore; use DPFrontend for "
                "data_parallel_size>1")
        self.engine = EngineCore(
            model, tp_size=tp_size, dcp_size=dcp_size,
            enable_expert_parallel=enable_ep,
            pp_size=pp_size, master_port=master_port,
            config_kwargs=config_kwargs, gpus=gpus,
            pipeline_credit=pipeline_credit,
            pipeline_mode=pipeline_mode,
            pipeline_microbatch_tokens=pipeline_microbatch_tokens,
            max_concurrent_batches=max_concurrent_batches,
            runner_cls=runner_cls, transport_cls=transport_cls)
        self.states = {}
        self._seqs = {}
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._commands = queue.SimpleQueue()
        self._ids = itertools.count()
        self._slots = threading.BoundedSemaphore(self.max_pending_requests)
        self._accepting = False
        self._counts = dict(completed=0, aborted=0, errors=0, rejected=0)

    def add(self, prompt: str | list[int], sampling_params):
        """Return a request handle without waiting for an in-flight GPU step."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt = list(prompt)
        if (not prompt or sampling_params.max_tokens < 1
                or len(prompt) + sampling_params.max_tokens > self.engine.config.max_model_len):
            raise ValueError("prompt plus completion must fit max_model_len, with positive lengths")
        if any(type(t) is not int or t < 0 for t in prompt):
            raise ValueError("prompt IDs must be nonnegative integers")
        with self._lock:
            if not self._accepting:
                raise RuntimeError("Engine is not accepting requests")
            if not self._slots.acquire(blocking=False):
                self._counts["rejected"] += 1
                raise EngineOverloaded("Request admission limit reached")
            rid, state = next(self._ids), _StreamState(self.stream_buffer_tokens)
            self.states[rid] = state
            self._commands.put(("add", rid, (prompt, sampling_params)))
        self._wake.set()
        return rid, state

    def abort(self, request_id):
        """Cancel at the next step boundary; scheduler has exactly one owner."""
        with self._lock:
            if request_id not in self.states:
                return
            self._commands.put(("abort", request_id, None))
        self._wake.set()

    def stats(self):
        with self._lock:
            return dict(self._counts, active=len(self.states), accepting=self._accepting)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("AsyncEngine has already been started")
        if self._stop.is_set():
            raise RuntimeError("A stopped EngineCore cannot be restarted")
        self._accepting = True
        self._thread = threading.Thread(target=self._loop,
                                         name="vllm-learn-engine",
                                         daemon=True)
        self._thread.start()

    def stop(self):
        with self._lock:
            self._accepting = False
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=getattr(self.engine, '_worker_timeout', 10))
            if self._thread.is_alive():
                raise TimeoutError("Engine step has not returned; shutdown stays with the owner thread")
        else:
            self.engine.shutdown()

    def _finish(self, rid, state, reason, error=None):
        state.close(reason, error)
        self._seqs.pop(rid, None)
        with self._lock:
            if self.states.pop(rid, None) is not None:
                self._slots.release()
                key = "errors" if error else ("aborted" if reason == "cancelled" else "completed")
                self._counts[key] += 1

    def _drain_commands(self):
        # A finite boundary keeps sustained arrivals from starving forward.
        for _ in range(self.max_pending_requests * 2):
            try:
                kind, rid, payload = self._commands.get_nowait()
            except queue.Empty:
                break
            state = self.states.get(rid)
            if state is None:
                continue
            if kind == "add":
                self._seqs[rid] = self.engine.add_request(*payload)
                state.times["enqueued"] = time.perf_counter()
            else:
                seq = self._seqs.get(rid)
                if seq is not None:
                    self.engine.abort_request(seq.seq_id)
                self._finish(rid, state, "cancelled")

    def _publish(self):
        """Publish all newly generated tokens and close finished streams."""
        for seq_id, seq in list(self._seqs.items()):
            state = self.states.get(seq_id)
            if state is None:
                continue
            tokens = seq.completion_token_ids
            while state.sent < len(tokens):
                if state.q.qsize() >= state.capacity:
                    self.engine.abort_request(seq.seq_id)
                    self._finish(seq_id, state, "error", RuntimeError("Slow consumer exceeded stream buffer"))
                    break
                state.times.setdefault("first_token_published", time.perf_counter())
                state.times.setdefault("token_published", []).append(time.perf_counter())
                state.q.put_nowait(tokens[state.sent])
                state.sent += 1
            if seq.is_finished and not state.finished:
                reason = (
                    "stop" if not seq.ignore_eos and
                    seq.completion_token_ids and
                    seq.completion_token_ids[-1] == self.engine.config.eos
                    else "length")
                self._finish(seq_id, state, reason)

    def _loop(self):
        failure = None
        try:
            while not self._stop.is_set():
                self._wake.clear()
                self._drain_commands()
                if self.engine.is_finished():
                    has_work = False
                else:
                    self.engine.step()
                    self._publish()
                    has_work = self.engine.last_step_had_work
                if not has_work:
                    self._wake.wait(timeout=0.01)
        except Exception as exc:
            failure = exc
        finally:
            with self._lock:
                self._accepting = False
                pending = list(self.states.items())
            for rid, state in pending:
                seq = self._seqs.get(rid)
                if seq is not None:
                    self.engine.abort_request(seq.seq_id)
                self._finish(rid, state, "error" if failure else "cancelled", failure)
            self._stop.set()
            self.engine.shutdown()

    def generate(self, prompt, sampling_params):
        _, state = self.add(prompt, sampling_params)
        tokens = []
        while True:
            token = state.q.get()
            if token is None:
                if state.error is not None:
                    raise state.error
                return tokens, state.finish_reason
            tokens.append(token)
