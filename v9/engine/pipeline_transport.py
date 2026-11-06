"""FIFO PP activation transport; shape/dtype come from scheduler messages.

All stages receive the same ordered batch descriptions. Only hidden/residual
travel over P2P: no GPU metadata header, device-to-host copy or lane protocol.
"""
from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class _SendRecord:
    requests: tuple
    tensors: tuple[torch.Tensor, torch.Tensor]
    ready_event: object = None


class PipelineTransport:
    """One ordered link per stage; the Core batch queue supplies concurrency."""

    def __init__(self, pp_size, pp_rank, inner_size, credit=4,
                 async_enabled=True, device="cuda", recv_group=None,
                 send_group=None, rank_offset=0):
        if pp_size < 1 or not 0 <= pp_rank < pp_size or inner_size < 1 or credit < 1:
            raise ValueError("invalid PP coordinates or send credit")
        self.pp_rank = pp_rank
        self.credit = credit
        self.async_enabled = async_enabled
        self.recv_group, self.send_group = recv_group, send_group
        self.device = torch.device(device)
        rank = dist.get_rank() if dist.is_initialized() else 0
        inner_rank = rank % inner_size
        self.src = (rank_offset + (pp_rank - 1) * inner_size + inner_rank
                    if pp_rank > 0 else None)
        self.dst = (rank_offset + (pp_rank + 1) * inner_size + inner_rank
                    if pp_rank + 1 < pp_size else None)
        self._sends = deque()
        self._last_batch_id = -1
        self.comm_stream = (torch.cuda.Stream(device=self.device)
                            if self.device.type == "cuda" else None)

    def _check_batch(self, batch_id):
        # A per-request numerical probe can execute several forwards for one
        # batch. Equal IDs are valid; moving backwards violates FIFO ordering.
        if batch_id < 0 or batch_id < self._last_batch_id:
            raise ValueError(f"PP batch order moved backwards: {self._last_batch_id} -> {batch_id}")
        self._last_batch_id = batch_id

    def _wait_oldest(self):
        record = self._sends.popleft()
        for request in record.requests:
            request.wait()

    def send_forward(self, hidden, residual, *, microbatch_id):
        if self.dst is None:
            return
        self._check_batch(microbatch_id)
        if (hidden.ndim != 2 or hidden.shape != residual.shape
                or hidden.dtype != residual.dtype or hidden.device != residual.device):
            raise ValueError("PP hidden/residual must have the same 2D shape, dtype and device")
        if len(self._sends) >= self.credit:
            self._wait_oldest()
        hidden, residual = hidden.contiguous(), residual.contiguous()
        if not self.async_enabled:
            dist.send(hidden, dst=self.dst, group=self.send_group)
            dist.send(residual, dst=self.dst, group=self.send_group)
            return
        ready = None
        if self.comm_stream is None:
            requests = tuple(dist.isend(t, dst=self.dst, group=self.send_group)
                             for t in (hidden, residual))
        else:
            ready = torch.cuda.Event()
            torch.cuda.current_stream(hidden.device).record_event(ready)
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(ready)
                requests = tuple(dist.isend(t, dst=self.dst, group=self.send_group)
                                 for t in (hidden, residual))
        self._sends.append(_SendRecord(requests, (hidden, residual), ready))

    def recv_forward(self, *, microbatch_id, shape, dtype):
        if self.src is None:
            raise RuntimeError("stage 0 cannot receive PP activations")
        self._check_batch(microbatch_id)
        if len(shape) != 2 or shape[0] < 0 or shape[1] < 1:
            raise ValueError(f"invalid scheduled PP shape {shape}")
        hidden = torch.empty(shape, dtype=dtype, device=self.device)
        residual = torch.empty_like(hidden)
        if not self.async_enabled:
            dist.recv(hidden, src=self.src, group=self.recv_group)
            dist.recv(residual, src=self.src, group=self.recv_group)
        else:
            if self.comm_stream is None:
                requests = tuple(dist.irecv(t, src=self.src, group=self.recv_group)
                                 for t in (hidden, residual))
            else:
                with torch.cuda.stream(self.comm_stream):
                    requests = tuple(dist.irecv(t, src=self.src, group=self.recv_group)
                                     for t in (hidden, residual))
            # NCCL Work.wait establishes dependencies on the current compute
            # stream. Gloo waits on CPU. Neither reads GPU tensor values.
            for request in requests:
                request.wait()
        return hidden, residual

    def drain(self):
        while self._sends:
            self._wait_oldest()

    def shutdown(self):
        self.drain()
