"""Qwen3-Omni PP activation transport.

The protocol is a four-int header [tokens, hidden, microbatch_id, lane]
followed by hidden and residual. Headers are posted lazily to avoid a NCCL
startup rendezvous before stage 0 has loaded its model. Payloads and sends are
asynchronous and use a dedicated communication stream when CUDA is available.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

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
    ready_event: object = None


class PipelineTransport:
    """Point-to-point activation transport for adjacent PP stages.

    The Python worker still processes one model message at a time. The
    transport itself does not block the sender: stage i can compute the next
    microbatch while the previous activation is in flight to stage i+1.
    """

    def __init__(self, pp_size: int, pp_rank: int, inner_size: int,
                 lanes: int = 2, credit: int = 4,
                 async_enabled: bool = True,
                 device: str = "cuda", recv_group=None, send_group=None,
                 rank_offset: int = 0):
        self.pp_size = int(pp_size)
        self.pp_rank = int(pp_rank)
        self.inner_size = int(inner_size)
        self.lanes = max(1, int(lanes))
        self.credit = max(1, int(credit))
        self.async_enabled = bool(async_enabled)
        self.recv_group, self.send_group = recv_group, send_group
        self.device = torch.device(device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_inner_rank = self.rank % self.inner_size
        self.src = (rank_offset + (self.pp_rank - 1) * self.inner_size +
                    self.local_inner_rank if self.pp_rank > 0 else None)
        self.dst = (rank_offset + (self.pp_rank + 1) * self.inner_size +
                    self.local_inner_rank
                    if self.pp_rank + 1 < self.pp_size else None)

        self._recv: list[_RecvSlot | None] = [None] * self.lanes
        self._sends: list[deque[_SendRecord]] = [
            deque() for _ in range(self.lanes)]
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
            request = dist.irecv(shape, src=self.src, group=self.recv_group)
        else:
            with torch.cuda.stream(self.comm_stream):
                request = dist.irecv(shape, src=self.src, group=self.recv_group)
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
        ready_event = None

        if not self.async_enabled:
            dist.send(header, dst=self.dst, group=self.send_group)
            dist.send(hidden, dst=self.dst, group=self.send_group)
            dist.send(residual, dst=self.dst, group=self.send_group)
            return

        if self.comm_stream is None:
            requests = (
                dist.isend(header, dst=self.dst, group=self.send_group),
                dist.isend(hidden, dst=self.dst, group=self.send_group),
                dist.isend(residual, dst=self.dst, group=self.send_group),
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
                    dist.isend(header, dst=self.dst, group=self.send_group),
                    dist.isend(hidden, dst=self.dst, group=self.send_group),
                    dist.isend(residual, dst=self.dst, group=self.send_group),
                )

        pending.append(_SendRecord(
            requests=tuple(requests),
            tensors=(header, hidden, residual),
            ready_event=ready_event,
        ))

    @staticmethod
    def _validate_header(tokens, hidden_size, microbatch_id, msg_lane, lane,
                         expected_microbatch_id, expected_shape):
        if msg_lane != lane:
            raise RuntimeError(f"PP lane mismatch: expected {lane}, received {msg_lane}")
        if tokens < 0 or hidden_size <= 0:
            raise RuntimeError(f"invalid PP activation shape {(tokens, hidden_size)}")
        if expected_microbatch_id is not None and microbatch_id != expected_microbatch_id:
            raise RuntimeError(
                f"PP batch mismatch: expected {expected_microbatch_id}, received {microbatch_id}")
        if expected_shape is not None and (tokens, hidden_size) != tuple(expected_shape):
            raise RuntimeError(
                f"PP shape mismatch: expected {expected_shape}, received {(tokens, hidden_size)}")

    def recv_forward(self, lane: int = 0, *, expected_microbatch_id=None,
                     expected_shape=None, dtype=torch.bfloat16):
        if self.src is None:
            raise RuntimeError("stage 0 cannot recv_forward")
        lane %= self.lanes

        if not self.async_enabled:
            shape = torch.empty(4, dtype=torch.int64, device=self.device)
            dist.recv(shape, src=self.src, group=self.recv_group)
            tokens, hidden_size, microbatch_id, msg_lane = map(
                int, shape.cpu().tolist())
            self._validate_header(tokens, hidden_size, microbatch_id, msg_lane,
                                  lane, expected_microbatch_id, expected_shape)
            hidden = torch.empty((tokens, hidden_size), dtype=dtype,
                                 device=self.device)
            residual = torch.empty_like(hidden)
            dist.recv(hidden, src=self.src, group=self.recv_group)
            dist.recv(residual, src=self.src, group=self.recv_group)
            return hidden, residual, microbatch_id, msg_lane

        slot = self._recv[lane]
        if slot is None:
            self._post_shape(lane)
            slot = self._recv[lane]
        self._wait(slot.request)
        self._recv[lane] = None
        tokens, hidden_size, microbatch_id, msg_lane = map(
            int, slot.shape.cpu().tolist())
        self._validate_header(tokens, hidden_size, microbatch_id, msg_lane,
                              lane, expected_microbatch_id, expected_shape)

        hidden = torch.empty((tokens, hidden_size), dtype=dtype,
                             device=self.device)
        residual = torch.empty_like(hidden)
        if self.comm_stream is None:
            requests = (dist.irecv(hidden, src=self.src, group=self.recv_group),
                        dist.irecv(residual, src=self.src, group=self.recv_group))
        else:
            with torch.cuda.stream(self.comm_stream):
                requests = (dist.irecv(hidden, src=self.src, group=self.recv_group),
                            dist.irecv(residual, src=self.src, group=self.recv_group))
        # The worker loop is single-message-at-a-time. Posting another header
        # here would leave an unmatched irecv after the final message and can
        # make NCCL teardown hang. The next recv call posts its header before
        # waiting, retaining lazy overlap without orphaned work.
        self._wait_payload(requests, hidden.device)
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
        """Compatibility alias for draining outstanding sends."""
        self.drain()
