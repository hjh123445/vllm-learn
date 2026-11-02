"""Qwen3-Omni PP activation transport.

The old PP path used ``dist.recv``/``dist.send`` directly inside the model
forward.  That made every stage a single blocking mailbox.  This transport
keeps two lane mailboxes, pre-posts the next shape receive, and uses
``isend``/``irecv`` with a small credit window.  It deliberately stays in
Python/NCCL so that correctness can be checked before introducing a custom
CUDA P2P kernel.

The lane is a scheduling key, not a second copy of model state.  A lane must
receive messages in order; different lanes may be in flight concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import torch
import torch.distributed as dist


@dataclass
class _RecvSlot:
    shape: torch.Tensor
    request: object


@dataclass
class _SendRecord:
    requests: list
    tensors: tuple[torch.Tensor, ...]
    lane: int
    microbatch_id: int
    started: float


class PipelineTransport:
    """Point-to-point activation transport for adjacent PP stages.

    ``lanes`` defaults to two (DualPipe-style forward lanes).  The protocol is
    shape header ``[tokens, hidden, microbatch_id, lane]`` followed by hidden
    and residual tensors.  Only adjacent ranks communicate, so TP/EP groups
    remain independent.
    """

    def __init__(self, pp_size: int, pp_rank: int, inner_size: int,
                 lanes: int = 2, credit: int = 4, device: str = "cuda"):
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.inner_size = inner_size
        self.lanes = max(1, int(lanes))
        self.credit = max(1, int(credit))
        self.device = torch.device(device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_inner_rank = self.rank % inner_size
        self.src = ((pp_rank - 1) * inner_size + self.local_inner_rank
                    if pp_rank > 0 else None)
        self.dst = ((pp_rank + 1) * inner_size + self.local_inner_rank
                    if pp_rank + 1 < pp_size else None)
        self._recv: list[_RecvSlot | None] = [None] * self.lanes
        self._sends: list[list[_SendRecord]] = [[] for _ in range(self.lanes)]
        self.trace: list[dict] = []
        self.comm_stream = (torch.cuda.Stream(device=self.device)
                            if self.device.type == "cuda" and torch.cuda.is_available()
                            else None)
        if self.src is not None:
            for lane in range(self.lanes):
                self._post_shape(lane)

    def _post_shape(self, lane: int):
        shape = torch.empty(4, dtype=torch.int64, device=self.device)
        req = dist.irecv(shape, src=self.src)
        self._recv[lane] = _RecvSlot(shape, req)

    def _wait_oldest(self, lane: int):
        pending = self._sends[lane]
        if not pending:
            return
        rec = pending.pop(0)
        for req in rec.requests:
            req.wait()
        self.trace.append({"kind": "send", "lane": rec.lane,
                           "microbatch_id": rec.microbatch_id,
                           "start": rec.started, "end": time.perf_counter()})

    def send_forward(self, hidden: torch.Tensor, residual: torch.Tensor,
                     lane: int = 0, microbatch_id: int = 0):
        if self.dst is None:
            return
        lane %= self.lanes
        hidden = hidden.contiguous()
        residual = residual.contiguous()
        header = torch.tensor([hidden.shape[0], hidden.shape[1], microbatch_id, lane],
                              dtype=torch.int64, device=hidden.device)
        started = time.perf_counter()
        if self.comm_stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(ready)
                reqs = [dist.isend(header, dst=self.dst),
                        dist.isend(hidden, dst=self.dst),
                        dist.isend(residual, dst=self.dst)]
        else:
            reqs = [dist.isend(header, dst=self.dst),
                    dist.isend(hidden, dst=self.dst),
                    dist.isend(residual, dst=self.dst)]
        self._sends[lane].append(_SendRecord(reqs, (header, hidden, residual),
                                             lane, microbatch_id, started))
        if len(self._sends[lane]) >= self.credit:
            self._wait_oldest(lane)

    def recv_forward(self, lane: int = 0):
        if self.src is None:
            raise RuntimeError("stage 0 cannot recv_forward")
        lane %= self.lanes
        slot = self._recv[lane]
        assert slot is not None
        slot.request.wait()
        tokens, hidden_size, microbatch_id, msg_lane = map(int, slot.shape.cpu().tolist())
        # Arm the next receive before waiting for this payload.  This is the
        # key difference from the old one-message blocking mailbox.
        self._post_shape(lane)
        hidden = torch.empty((tokens, hidden_size), dtype=torch.bfloat16,
                             device=self.device)
        residual = torch.empty_like(hidden)
        started = time.perf_counter()
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                reqs = [dist.irecv(hidden, src=self.src),
                        dist.irecv(residual, src=self.src)]
        else:
            reqs = [dist.irecv(hidden, src=self.src),
                    dist.irecv(residual, src=self.src)]
        for req in reqs:
            req.wait()
        if self.comm_stream is not None:
            torch.cuda.current_stream(hidden.device).wait_stream(self.comm_stream)
        self.trace.append({"kind": "recv", "lane": msg_lane,
                           "microbatch_id": microbatch_id,
                           "start": started, "end": time.perf_counter()})
        return hidden, residual, microbatch_id, msg_lane

    def drain(self):
        for lane in range(self.lanes):
            while self._sends[lane]:
                self._wait_oldest(lane)
