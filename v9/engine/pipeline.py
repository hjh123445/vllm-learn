"""Pipeline-parallel transport and scheduling metadata."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class PipelinePlan:
    """One immutable unit sent through all PP stages."""

    microbatch_id: int
    virtual_engine: int
    data_parallel_rank: int
    is_prefill: bool
    message: dict
    sequences: tuple


@dataclass
class VirtualEngineState:
    """Bookkeeping for one virtual pipeline slot."""

    slot_id: int
    in_flight: int = 0


class PipelineTransport:
    """Stage-to-stage hidden-state transport for one PP lane.

    The shape header and activations use nonblocking sends.  A bounded queue
    keeps CUDA buffers alive until NCCL has consumed them, allowing the worker
    to receive and compute the next microbatch while the previous transfer is
    in flight.  ``flush`` is called during worker shutdown.
    """

    def __init__(self, pp_size: int, pp_rank: int, inner_size: int):
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.inner_size = inner_size
        self.enabled = pp_size > 1
        self._pending_sends = []
        self._max_pending = max(2, pp_size)

    @property
    def is_first_stage(self):
        return self.pp_rank == 0

    @property
    def is_last_stage(self):
        return self.pp_rank == self.pp_size - 1

    def _peer(self, delta: int) -> int:
        return dist.get_rank() + delta * self.inner_size

    def recv_forward(self, *, device, dtype):
        if not self.enabled or self.is_first_stage:
            return None
        src = self._peer(-1)
        shape = torch.empty(2, dtype=torch.int64, device=device)
        dist.recv(shape, src=src)
        tokens, hidden = (int(v) for v in shape.tolist())
        x = torch.empty((tokens, hidden), dtype=dtype, device=device)
        residual = torch.empty_like(x)
        dist.recv(x, src=src)
        dist.recv(residual, src=src)
        return x, residual

    def send_forward(self, x: torch.Tensor, residual: torch.Tensor):
        if not self.enabled or self.is_last_stage:
            return
        shape = torch.tensor((x.shape[0], x.shape[1]), dtype=torch.int64,
                             device=x.device)
        dst = self._peer(1)
        x_send = x.contiguous()
        residual_send = residual.contiguous()
        works = [dist.isend(shape, dst=dst),
                 dist.isend(x_send, dst=dst),
                 dist.isend(residual_send, dst=dst)]
        # Keep every send buffer referenced until its Work completes.
        self._pending_sends.append((works, (shape, x_send, residual_send)))
        if len(self._pending_sends) > self._max_pending:
            self._retire_one()

    def _retire_one(self):
        works, _buffers = self._pending_sends.pop(0)
        for work in works:
            work.wait()

    def flush(self):
        while self._pending_sends:
            self._retire_one()
