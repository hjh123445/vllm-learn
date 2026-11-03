"""Online adapter for the v9 EngineCore.

The HTTP layer owns request queues; EngineCore remains the single scheduler
owner and is advanced by one background thread.  Keeping this adapter on the
new EngineCore path makes the server exercise the same worker/MQ/PP code as
offline inference instead of silently falling back to the legacy LLMEngine.
"""

import queue
import threading


class _StreamState:
    """A request-local token stream (engine thread writes, client reads)."""

    __slots__ = ("q", "sent", "finished", "finish_reason", "error")

    def __init__(self):
        self.q = queue.Queue()
        self.sent = 0
        self.finished = False
        self.finish_reason = None
        self.error = None

    def close(self):
        self.q.put(None)


class AsyncEngine:
    """Background-thread facade over :class:`engine.core.EngineCore`.

    ``tensor_parallel_size`` and ``expert_parallel_size`` are translated to
    EngineCore's ``tp_size``/``ep_size`` names for compatibility with the
    existing server command line and examples.
    """

    def __init__(self, model: str, **kwargs):
        from transformers import AutoTokenizer
        from engine.core import EngineCore

        tp_size = kwargs.pop("tp_size", kwargs.pop("tensor_parallel_size", 1))
        ep_size = kwargs.pop("ep_size", kwargs.pop("expert_parallel_size", 1))
        pp_size = kwargs.pop("pp_size", kwargs.pop("pipeline_parallel_size", 1))
        dcp_size = kwargs.pop("dcp_size", 1)
        dp_size = kwargs.pop("data_parallel_size", 1)
        gpus = kwargs.pop("gpus", None)
        master_port = kwargs.pop("master_port", 29600)
        pipeline_lanes = kwargs.pop("pipeline_lanes", 2)
        pipeline_credit = kwargs.pop("pipeline_credit", 4)
        pipeline_mode = kwargs.pop("pipeline_mode", "optimized")
        pipeline_microbatch_tokens = kwargs.pop("pipeline_microbatch_tokens", 2048)
        pipeline_stage_map = kwargs.pop("pipeline_stage_map", None)
        config_fields = {
            "max_num_batched_tokens", "max_num_seqs", "max_model_len",
            "gpu_memory_utilization", "enforce_eager", "kvcache_block_size",
            "pipeline_microbatches", "virtual_engine_size", "pipeline_schedule",
        }
        config_kwargs = {k: v for k, v in kwargs.items()
                         if k in config_fields}
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if dp_size > 1:
            raise ValueError(
                "AsyncEngine serves one EngineCore; use DPFrontend for "
                "data_parallel_size>1")
        self.engine = EngineCore(
            model, tp_size=tp_size, ep_size=ep_size, dcp_size=dcp_size,
            pp_size=pp_size, master_port=master_port,
            config_kwargs=config_kwargs, gpus=gpus,
            pipeline_lanes=pipeline_lanes, pipeline_credit=pipeline_credit,
            pipeline_mode=pipeline_mode,
            pipeline_microbatch_tokens=pipeline_microbatch_tokens,
            pipeline_stage_map=pipeline_stage_map)
        self.states = {}
        self._seqs = {}
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def add(self, prompt: str | list[int], sampling_params):
        """Submit a request and return ``(seq_id, stream_state)``."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        with self._lock:
            seq = self.engine.add_request(prompt, sampling_params)
            state = _StreamState()
            self.states[seq.seq_id] = state
            self._seqs[seq.seq_id] = seq
        self._wake.set()
        return seq.seq_id, state

    def start(self):
        if self._thread is not None:
            raise RuntimeError("AsyncEngine has already been started")
        self._thread = threading.Thread(target=self._loop,
                                         name="vllm-learn-engine",
                                         daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        with self._lock:
            self.engine.shutdown()
            self._thread = None

    def _publish(self):
        """Publish all newly generated tokens and close finished streams."""
        for seq_id, seq in list(self._seqs.items()):
            state = self.states.get(seq_id)
            if state is None:
                continue
            tokens = seq.completion_token_ids
            while state.sent < len(tokens):
                state.q.put(tokens[state.sent])
                state.sent += 1
            if seq.is_finished and not state.finished:
                state.finished = True
                state.finish_reason = (
                    "stop" if not seq.ignore_eos and
                    seq.completion_token_ids and
                    seq.completion_token_ids[-1] == self.engine.config.eos
                    else "length")
                state.close()

    def _loop(self):
        while not self._stop.is_set():
            try:
                with self._lock:
                    if self.engine.is_finished():
                        has_work = False
                    else:
                        self.engine.step()
                        self._publish()
                        has_work = self.engine.last_step_had_work
            except Exception as exc:
                for state in self.states.values():
                    if not state.finished:
                        state.error = exc
                        state.finished = True
                        state.finish_reason = "error"
                        state.close()
                self._stop.set()
                break
            if not has_work:
                self._wake.wait(timeout=0.01)
                self._wake.clear()

    def generate(self, prompt, sampling_params):
        _, state = self.add(prompt, sampling_params)
        tokens = []
        while True:
            token = state.q.get()
            if token is None:
                return tokens, state.finish_reason
