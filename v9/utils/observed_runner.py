"""Explicit experiment adapters; production engine code never imports this module.

Select these classes from a capture/probe script via runner_cls/transport_cls.
Environment variables alone do not instrument a normal EngineCore.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import torch

from engine.model_runner import ModelRunner
from engine.pipeline_transport import PipelineTransport
from layers.parallel import STATE
from utils.module_profiling import ModuleProfileRanges


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class ObservedPipelineTransport(PipelineTransport):
    """Host call intervals only, never a measurement of GPU transfer duration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []
        self._enabled = bool(os.getenv("VLLM_LEARN_TRACE_DIR"))

    def _observe(self, kind, operation, lane, batch_id, *args, **kwargs):
        if not self._enabled:
            return operation(*args, **kwargs)
        started = time.perf_counter()
        try:
            return operation(*args, **kwargs)
        finally:
            self.events.append(dict(name="pp." + kind, cat="p2p", ph="X",
                ts=started * 1e6, dur=(time.perf_counter() - started) * 1e6,
                args=dict(stage=self.pp_rank, lane=lane, microbatch_id=batch_id,
                          variant="optimized" if self.async_enabled else "baseline")))

    def send_forward(self, hidden, residual, lane=0, microbatch_id=0):
        return self._observe("send", super().send_forward, lane, microbatch_id,
                             hidden, residual, lane, microbatch_id)

    def recv_forward(self, lane=0, *, expected_microbatch_id=None,
                     expected_shape=None, dtype=torch.bfloat16):
        return self._observe("recv", super().recv_forward, lane, expected_microbatch_id,
            lane, expected_microbatch_id=expected_microbatch_id,
            expected_shape=expected_shape, dtype=dtype)


class _WorkerCapture:
    def __init__(self, runner):
        self.runner = runner
        self.stage = STATE.pp_rank
        local_size = STATE.pp_size * STATE.tp_size * STATE.cp_size
        self.rank = STATE.replica_rank * local_size + (
            STATE.pp_rank * STATE.tp_size + STATE.tp_rank) * STATE.cp_size + STATE.cp_rank
        self.world_rank = ((STATE.dp_rank * STATE.pp_size + STATE.pp_rank)
                           * STATE.tp_size + STATE.tp_rank) * STATE.cp_size + STATE.cp_rank
        self.trace_dir = os.getenv("VLLM_LEARN_TRACE_DIR")
        self.window = os.getenv("VLLM_LEARN_PROFILE_MODE", "scheduled") == "window"
        self.start_message = max(1, int(os.getenv("VLLM_LEARN_PROFILE_START_MESSAGE", "1")))
        self.limit = max(1, int(os.getenv("VLLM_LEARN_PROFILE_MESSAGES", "4")))
        self.messages = []
        self.events = []
        self.index = 0
        self.open = False
        self.cuda = torch.cuda.is_available()
        self.profiler = None
        self.audit = None
        self.flush_status = None
        self.ranges = ModuleProfileRanges(runner.model, stage=self.stage)
        if os.getenv("VLLM_LEARN_CUDA_PROFILE", "0") == "1":
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities,
                schedule=None if self.window else torch.profiler.schedule(wait=1, warmup=1,
                    active=int(os.getenv("VLLM_LEARN_PROFILE_STEPS", "2")), repeat=1),
                on_trace_ready=self._export,
                record_shapes=os.getenv("VLLM_LEARN_PROFILE_RECORD_SHAPES", "1") == "1",
                with_stack=False, profile_memory=False)
            if not self.window:
                self._start()

    def _export(self, profiler):
        directory = Path(self.trace_dir or "/tmp")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"cuda_rank{self.rank}.json"
        profiler.export_chrome_trace(str(path))
        from utils.trace_audit import audit_trace
        self.audit = audit_trace(path)
        _write(directory / f"audit_rank{self.rank}.json", self.audit)

    def _start(self):
        self.profiler.__enter__()
        self.open = True
        self.ranges.install()

    def _stop(self):
        if not self.open:
            return
        if self.cuda:
            torch.cuda.synchronize()
            from utils.cupti_flush import flush_cuda_activity
            self.flush_status = flush_cuda_activity()
        self.ranges.remove()
        try:
            self.profiler.__exit__(None, None, None)
        finally:
            self.open = False

    def before(self, seqs, prefill, lane, batch):
        self.index += 1
        if self.profiler is not None and self.window and self.index == self.start_message:
            if self.cuda:
                torch.cuda.synchronize()
            self._start()
        self.current = dict(message_index=self.index, microbatch_id=batch,
            lane=lane, is_prefill=prefill, scheduled_sequences=len(seqs), stage=self.stage)
        if self.open:
            return torch.profiler.record_function(
                f"rank{self.rank}.stage{self.stage}.message{self.index}.mb{batch}."
                f"{'prefill' if prefill else 'decode'}")
        return nullcontext()

    def after(self, started):
        if self.trace_dir:
            self.events.append(dict(name=f"stage{self.stage}.forward", cat="forward", ph="X",
                ts=started * 1e6, dur=(time.perf_counter() - started) * 1e6,
                pid=self.rank, tid="compute", args=self.current.copy()))
        if self.open:
            if self.window:
                self.messages.append(self.current.copy())
                if len(self.messages) >= self.limit:
                    self._stop()
            else:
                if self.cuda:
                    torch.cuda.synchronize()
                    from utils.cupti_flush import flush_cuda_activity
                    self.flush_status = flush_cuda_activity()
                self.profiler.step()

    def close(self):
        self._stop()
        if self.profiler is not None and self.window:
            _write(Path(self.trace_dir or "/tmp") / f"capture_rank{self.rank}.json",
                dict(rank=self.rank, world_rank=self.world_rank, mode="window",
                     start_message=self.start_message, requested_messages=self.limit,
                     captured_messages=self.messages, cuda_activities=self.cuda,
                     cuda_verified=bool(self.audit and self.audit['cuda_verified']),
                     cupti_flush=self.flush_status,
                     measurement="profile-only; synchronization/export distort throughput"))
        if self.trace_dir:
            events = list(self.events)
            for event in getattr(STATE.pipeline_transport, "events", []):
                events.append(dict(event, pid=self.rank, tid="comm"))
            _write(Path(self.trace_dir) / f"pipeline_rank{self.rank}.json",
                dict(displayTimeUnit="ms", traceEvents=events,
                     metadata=dict(rank=self.rank, world_rank=self.world_rank, pp_stage=self.stage,
                         clock="host_monotonic_us", measurement="CPU calls, not GPU duration",
                         tp_rank=STATE.tp_rank, ep_rank=STATE.ep_rank,
                         layer_range=[self.runner.model.model.pp_start, self.runner.model.model.pp_end])))


class ObservedModelRunner(ModelRunner):
    """Opt-in profiling, memory export, activation and numerical diagnostics."""

    def __init__(self, config, dtype=None):
        self._observation = None  # Base initialization executes warmup via self.run.
        self._current = None
        self._handles = []
        self._closed = False
        super().__init__(config, dtype)
        self._logit_dir = os.getenv("VLLM_LEARN_LOGITS_DIR")
        self._logit_limit = int(os.getenv("VLLM_LEARN_LOGITS_LIMIT", "128"))
        self._logit_count = 0
        if self._logit_dir and STATE.tp_rank == STATE.cp_rank == 0:
            self._handles.append(self.sampler.register_forward_pre_hook(self._capture_logits))
        directory = os.getenv("VLLM_LEARN_ACTIVATIONS_DIR")
        if directory and STATE.tp_rank == STATE.cp_rank == 0:
            from utils.activation_capture import install_activation_capture
            if STATE.replica_size > 1:
                directory = str(Path(directory) / f"dp{STATE.replica_rank}")
            self._handles.extend(install_activation_capture(self.model, directory, STATE.pp_rank))
        if os.getenv("VLLM_LEARN_TRACE_DIR") or os.getenv("VLLM_LEARN_CUDA_PROFILE", "0") == "1":
            self._observation = _WorkerCapture(self)

    def run(self, seqs, is_prefill=True, pipeline_lane=0, microbatch_id=0):
        previous = self._current
        self._current = (seqs, is_prefill)
        try:
            if self._observation is None or previous is not None:
                return super().run(seqs, is_prefill, pipeline_lane, microbatch_id)
            scope = self._observation.before(seqs, is_prefill, pipeline_lane, microbatch_id)
            started = time.perf_counter()
            try:
                with scope:
                    return super().run(seqs, is_prefill, pipeline_lane, microbatch_id)
            finally:
                self._observation.after(started)
        finally:
            self._current = previous

    def _capture_logits(self, _sampler, inputs):
        seqs, is_prefill = self._current
        selected = [i for i, seq in enumerate(seqs)
                    if seq.num_cached_tokens + seq.num_scheduled_tokens >= len(seq)]
        if not selected:
            return
        if self._logit_count >= self._logit_limit:
            raise RuntimeError("Diagnostic logits limit exceeded; reduce probe size")
        path = Path(self._logit_dir)
        if STATE.replica_size > 1:
            path /= f"dp{STATE.replica_rank}"
        path.mkdir(parents=True, exist_ok=True)
        record = dict(logits=inputs[0][selected].detach().float().cpu(),
            seq_ids=[seqs[i].seq_id for i in selected], prompt_lengths=[len(seqs[i]) for i in selected],
            pp_rank=STATE.pp_rank, dp_rank=STATE.replica_rank, is_prefill=is_prefill,
            measurement="synchronizing numerical diagnostic; no throughput")
        torch.save(record, path / f"logits_stage{STATE.pp_rank}_{self._logit_count:04d}.pt")
        self._logit_count += 1

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._observation is not None:
                self._observation.close()
            directory = os.getenv("VLLM_LEARN_METRICS_DIR")
            if directory and torch.cuda.is_available():
                torch.cuda.synchronize()
                local_size = STATE.pp_size * STATE.tp_size * STATE.cp_size
                rank = STATE.replica_rank * local_size + (
                    STATE.pp_rank * STATE.tp_size + STATE.tp_rank) * STATE.cp_size + STATE.cp_rank
                _write(Path(directory) / f"memory_rank{rank}.json",
                    dict(rank=rank, gpu=torch.cuda.current_device(), pp_stage=STATE.pp_rank,
                         tp_rank=STATE.tp_rank, ep_rank=STATE.ep_rank,
                         allocated_bytes=torch.cuda.memory_allocated(),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         reserved_bytes=torch.cuda.memory_reserved(),
                         kv_cache_bytes=self.kv_cache.numel() * self.kv_cache.element_size(),
                         local_parameter_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters())))
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            super().shutdown()
