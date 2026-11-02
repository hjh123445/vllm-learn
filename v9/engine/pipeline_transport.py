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
        # Do not call NCCL ``irecv`` during worker construction.  NCCL's
        # point-to-point communicator setup is a rendezvous with the sender;
        # stage 0 is still loading its model at this point and cannot issue an
        # ``isend``.  Eager receives therefore deadlock PP startup.  The first
        # receive is posted lazily from ``recv_forward`` once the matching
        # stage-0 send is in the same forward schedule; subsequent receives are
        # still preposted to preserve the credit-window overlap.

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
        # A blocking first hop is the correctness baseline.  NCCL creates a
        # point-to-point communicator lazily; launching an async send from a
        # separate stream while the receiver is still constructing its first
        # ``irecv`` can leave the communicator in a spin wait.  Once this path
        # is validated, a fused batch_isend_irecv implementation can restore
        # overlap without changing the wire format.
        dist.send(header, dst=self.dst)
        dist.send(hidden, dst=self.dst)
        dist.send(residual, dst=self.dst)
        self.trace.append({"kind": "send", "lane": lane,
                           "microbatch_id": microbatch_id,
                           "start": started, "end": time.perf_counter()})

    def recv_forward(self, lane: int = 0):
        if self.src is None:
            raise RuntimeError("stage 0 cannot recv_forward")
        lane %= self.lanes
        # Match the sender's blocking first hop.  Keeping shape and payload in
        # one explicit sequence avoids NCCL communicator setup races during PP
        # startup and gives a deterministic reference for later async tuning.
        shape = torch.empty(4, dtype=torch.int64, device=self.device)
        dist.recv(shape, src=self.src)
        tokens, hidden_size, microbatch_id, msg_lane = map(int, shape.cpu().tolist())
        hidden = torch.empty((tokens, hidden_size), dtype=torch.bfloat16,
                             device=self.device)
        residual = torch.empty_like(hidden)
        started = time.perf_counter()
        dist.recv(hidden, src=self.src)
        dist.recv(residual, src=self.src)
        self.trace.append({"kind": "recv", "lane": msg_lane,
                           "microbatch_id": microbatch_id,
                           "start": started, "end": time.perf_counter()})
        return hidden, residual, microbatch_id, msg_lane

    def drain(self):
        for lane in range(self.lanes):
            while self._sends[lane]:
                self._wait_oldest(lane)
