"""Qwen3-Omni PP activation transport.

The protocol is a four-int header [tokens, hidden, microbatch_id, lane]
followed by hidden and residual. Headers are posted lazily to avoid a NCCL
startup rendezvous before stage 0 has loaded its model. Payloads and sends are
asynchronous and use a dedicated communication stream when CUDA is available.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import torch
import torch.distributed as dist


@dataclass
class _RecvSlot:
    shape: torch.Tensor
    request: object


@dataclass
class _SendRecord:
    requests: tuple
    tensors: tuple[torch.Tensor, ...]
    lane: int
    microbatch_id: int
    started: float
    ready_event: object = None


class PipelineTransport:
    """Point-to-point activation transport for adjacent PP stages.

    The Python worker still processes one model message at a time. The
    transport itself does not block the sender: stage i can compute the next
    microbatch while the previous activation is in flight to stage i+1.
    """

    def __init__(self, pp_size: int, pp_rank: int, inner_size: int,
                 lanes: int = 2, credit: int = 4,
                 async_enabled: bool = True, trace_variant: str = "optimized",
                 device: str = "cuda"):
        self.pp_size = int(pp_size)
        self.pp_rank = int(pp_rank)
        self.inner_size = int(inner_size)
        self.lanes = max(1, int(lanes))
        self.credit = max(1, int(credit))
        self.async_enabled = bool(async_enabled)
        self.trace_variant = str(trace_variant)
        self.device = torch.device(device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_inner_rank = self.rank % self.inner_size
        self.src = ((self.pp_rank - 1) * self.inner_size +
                    self.local_inner_rank if self.pp_rank > 0 else None)
        self.dst = ((self.pp_rank + 1) * self.inner_size +
                    self.local_inner_rank
                    if self.pp_rank + 1 < self.pp_size else None)

        self._recv: list[_RecvSlot | None] = [None] * self.lanes
        self._sends: list[deque[_SendRecord]] = [
            deque() for _ in range(self.lanes)]
        self.trace: list[dict] = []
        self.trace_origin = time.perf_counter()
        self.comm_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )

    def _post_shape(self, lane: int):
        """Post the next header receive for a lane.

        This is deliberately called from recv_forward, after the worker has
        entered the corresponding scheduler message. Posting receives during
        model construction can deadlock NCCL initialization.
        """
        shape = torch.empty(4, dtype=torch.int64, device=self.device)
        if self.comm_stream is None:
            request = dist.irecv(shape, src=self.src)
        else:
            with torch.cuda.stream(self.comm_stream):
                request = dist.irecv(shape, src=self.src)
        self._recv[lane] = _RecvSlot(shape, request)

    @staticmethod
    def _wait(request):
        request.wait()

    def _wait_oldest(self, lane: int):
        pending = self._sends[lane]
        if not pending:
            return
        record = pending.popleft()
        for request in record.requests:
            self._wait(request)
        self.trace.append({
            "kind": "send",
            "lane": record.lane,
            "microbatch_id": record.microbatch_id,
            "start": record.started,
            "end": time.perf_counter(),
            "stage": self.pp_rank,
            "variant": self.trace_variant,
        })

    def _wait_payload(self, requests, ready_device):
        for request in requests:
            self._wait(request)
        if self.comm_stream is None:
            return
        ready = torch.cuda.Event()
        with torch.cuda.stream(self.comm_stream):
            ready.record(self.comm_stream)
        torch.cuda.current_stream(ready_device).wait_event(ready)

    def send_forward(self, hidden: torch.Tensor, residual: torch.Tensor,
                     lane: int = 0, microbatch_id: int = 0):
        if self.dst is None:
            return
        lane %= self.lanes
        pending = self._sends[lane]
        if len(pending) >= self.credit:
            self._wait_oldest(lane)

        hidden = hidden.contiguous()
        residual = residual.contiguous()
        header = torch.tensor(
            [hidden.shape[0], hidden.shape[1], microbatch_id, lane],
            dtype=torch.int64, device=hidden.device)
        started = time.perf_counter()
        ready_event = None

        if not self.async_enabled:
            dist.send(header, dst=self.dst)
            dist.send(hidden, dst=self.dst)
            dist.send(residual, dst=self.dst)
            self.trace.append({"kind": "send", "lane": lane,
                               "microbatch_id": int(microbatch_id),
                               "start": started, "end": time.perf_counter(),
                               "stage": self.pp_rank, "variant": self.trace_variant})
            return

        if self.comm_stream is None:
            requests = (
                dist.isend(header, dst=self.dst),
                dist.isend(hidden, dst=self.dst),
                dist.isend(residual, dst=self.dst),
            )
        else:
            # The model writes hidden/residual on the current compute stream.
            # The communication stream must wait for that production event
            # before NCCL reads either tensor.
            ready_event = torch.cuda.Event()
            torch.cuda.current_stream(hidden.device).record_event(ready_event)
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(ready_event)
                requests = (
                    dist.isend(header, dst=self.dst),
                    dist.isend(hidden, dst=self.dst),
                    dist.isend(residual, dst=self.dst),
                )

        pending.append(_SendRecord(
            requests=tuple(requests),
            tensors=(header, hidden, residual),
            lane=lane,
            microbatch_id=int(microbatch_id),
            started=started,
            ready_event=ready_event,
        ))

    def recv_forward(self, lane: int = 0):
        if self.src is None:
            raise RuntimeError("stage 0 cannot recv_forward")
        lane %= self.lanes

        if not self.async_enabled:
            shape = torch.empty(4, dtype=torch.int64, device=self.device)
            started = time.perf_counter()
            dist.recv(shape, src=self.src)
            tokens, hidden_size, microbatch_id, msg_lane = map(
                int, shape.cpu().tolist())
            hidden = torch.empty((tokens, hidden_size), dtype=torch.bfloat16,
                                 device=self.device)
            residual = torch.empty_like(hidden)
            dist.recv(hidden, src=self.src)
            dist.recv(residual, src=self.src)
            self.trace.append({
                "kind": "recv", "lane": msg_lane,
                "microbatch_id": microbatch_id, "start": started,
                "end": time.perf_counter(), "stage": self.pp_rank,
                "variant": self.trace_variant,
            })
            return hidden, residual, microbatch_id, msg_lane

        slot = self._recv[lane]
        if slot is None:
            self._post_shape(lane)
            slot = self._recv[lane]
        self._wait(slot.request)
        self._recv[lane] = None
        tokens, hidden_size, microbatch_id, msg_lane = map(
            int, slot.shape.cpu().tolist())
        if msg_lane != lane:
            raise RuntimeError(
                f"PP lane mismatch: expected {lane}, received {msg_lane}")
        if tokens < 0 or hidden_size <= 0:
            raise RuntimeError(f"invalid PP activation shape {(tokens, hidden_size)}")

        hidden = torch.empty((tokens, hidden_size), dtype=torch.bfloat16,
                             device=self.device)
        residual = torch.empty_like(hidden)
        started = time.perf_counter()
        if self.comm_stream is None:
            requests = (dist.irecv(hidden, src=self.src),
                        dist.irecv(residual, src=self.src))
        else:
            with torch.cuda.stream(self.comm_stream):
                requests = (dist.irecv(hidden, src=self.src),
                            dist.irecv(residual, src=self.src))
        # The worker loop is single-message-at-a-time. Posting another header
        # here would leave an unmatched irecv after the final message and can
        # make NCCL teardown hang. The next recv call posts its header before
        # waiting, retaining lazy overlap without orphaned work.
        self._wait_payload(requests, hidden.device)
        self.trace.append({
            "kind": "recv", "lane": msg_lane,
            "microbatch_id": microbatch_id, "start": started,
            "end": time.perf_counter(), "stage": self.pp_rank,
            "variant": self.trace_variant,
        })
        return hidden, residual, microbatch_id, msg_lane

    def shutdown(self):
        """Release preposted receives and finish sends before NCCL teardown."""
        for lane, slot in enumerate(self._recv):
            if slot is None:
                continue
            abort = getattr(slot.request, "abort", None)
            if callable(abort):
                try:
                    abort()
                except Exception:
                    pass
            else:
                completed = getattr(slot.request, "is_completed", None)
                if callable(completed) and completed():
                    try:
                        slot.request.wait()
                    except Exception:
                        pass
            self._recv[lane] = None
        self.drain()
    def drain(self):
        """Wait for every outstanding send before process-group teardown."""
        for lane in range(self.lanes):
            while self._sends[lane]:
                self._wait_oldest(lane)

    def flush(self):
        """Compatibility alias used by ModelRunner shutdown/profiling hooks."""
        self.drain()
