"""Shared-memory control-plane channels for EngineCore workers.

Each endpoint serializes into a fixed shared-memory slot, or a temporary
shared segment for large media batches. A small
one-way notification pipe carries the byte count and a separate acknowledgement
pipe prevents the producer from overwriting the slot before the consumer has
copied it. This mirrors vLLM's shared-memory message queue at a compact,
Python-level scale. GPU tensor collectives remain in torch.distributed (NCCL).
"""

import multiprocessing as mp
import pickle
from multiprocessing.shared_memory import SharedMemory


SHUTDOWN = {"type": "shutdown"}
_DEFAULT_CAPACITY = 16 * 1024 * 1024
_MAX_PAYLOAD = 512 * 1024 * 1024


class SharedMemoryEndpoint:
    def __init__(self, shm_name, notify, ack, *, owner=False, sender=False,
                 capacity=_DEFAULT_CAPACITY):
        self.shm_name = shm_name
        self.notify = notify
        self.ack = ack
        self.owner = owner
        self.sender = sender
        self.capacity = capacity
        self._shm = None
        self._overflow = None

    def _buffer(self):
        if self._shm is None:
            self._shm = SharedMemory(name=self.shm_name)
        return self._shm.buf

    def send(self, message, *, wait_ack=True):
        if not self.sender:
            raise RuntimeError("shared-memory endpoint is receive-only")
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        if self._overflow is not None:
            raise RuntimeError('acknowledge the previous overflow payload before sending')
        if len(payload) > _MAX_PAYLOAD:
            raise ValueError(f'scheduler message exceeds {_MAX_PAYLOAD} byte transport limit')
        try:
            if len(payload) > self.capacity:
                # Real multimodal batches exceed a text-sized fixed slot.
                # Keep a temporary segment alive until this consumer ACKs
                # copying it; do not grow all idle/rank response slots.
                self._overflow = SharedMemory(create=True, size=len(payload))
                self._overflow.buf[:] = payload
                self.notify.send(('overflow', self._overflow.name, len(payload)))
            else:
                buf = self._buffer()
                buf[:len(payload)] = payload
                self.notify.send(len(payload))
        except BaseException:
            self._release_overflow()
            raise
        # One producer/consumer pair owns the slot. This acknowledgement is
        # what makes reuse safe without a heavyweight lock.
        if wait_ack:
            self.wait_ack()

    def wait_ack(self, wait=None):
        """Wait until the consumer has copied the current shared slot."""
        if wait is not None:
            wait(self.ack)
        self.ack.recv()
        self._release_overflow()

    def _release_overflow(self):
        segment, self._overflow = self._overflow, None
        if segment is not None:
            segment.close()
            try:
                segment.unlink()
            except FileNotFoundError:
                pass

    def recv(self):
        if self.sender:
            raise RuntimeError("shared-memory endpoint is send-only")
        notice = self.notify.recv()
        if isinstance(notice, tuple):
            tag, name, size = notice
            if tag != 'overflow' or not self.capacity < size <= _MAX_PAYLOAD:
                raise ValueError('invalid overflow notification')
            segment = SharedMemory(name=name)
            try:
                payload = bytes(segment.buf[:size])
            finally:
                segment.close()
        else:
            size = notice
            if not isinstance(size, int) or not 0 < size <= self.capacity:
                raise ValueError('invalid shared slot notification')
            payload = bytes(self._buffer()[:size])
        message = pickle.loads(payload)
        self.ack.send(True)
        return message

    def poll(self, timeout=0.0):
        if self.sender:
            raise RuntimeError("shared-memory endpoint is send-only")
        return self.notify.poll(timeout)

    def close(self):
        self._release_overflow()
        if self.notify is not None:
            self.notify.close()
            self.notify = None
        if self.ack is not None:
            self.ack.close()
            self.ack = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self.owner:
            try:
                shm = SharedMemory(name=self.shm_name)
                shm.unlink()
                shm.close()
            except FileNotFoundError:
                pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shm"] = None
        return state


def _channel_pair(reverse=False, capacity=_DEFAULT_CAPACITY):
    # notify: producer -> consumer; ack: consumer -> producer
    notify_recv, notify_send = mp.Pipe(duplex=False)
    ack_recv, ack_send = mp.Pipe(duplex=False)
    shm = SharedMemory(create=True, size=capacity)
    if reverse:
        # Parent receives, spawned child sends.
        parent = SharedMemoryEndpoint(shm.name, notify_recv, ack_send,
                                      owner=True, sender=False, capacity=capacity)
        child = SharedMemoryEndpoint(shm.name, notify_send, ack_recv,
                                     owner=False, sender=True, capacity=capacity)
        parent._shm = shm
        return parent, child
    parent = SharedMemoryEndpoint(shm.name, notify_send, ack_recv,
                                  owner=True, sender=True, capacity=capacity)
    child = SharedMemoryEndpoint(shm.name, notify_recv, ack_send,
                                 owner=False, sender=False, capacity=capacity)
    parent._shm = shm
    return parent, child


def make_channels(world_size: int):
    """Return shared-memory broadcast and response endpoint pairs."""
    bc = [_channel_pair() for _ in range(world_size)]
    rc = [_channel_pair(reverse=True) for _ in range(world_size)]
    return bc, rc


def broadcast(msg, bc_parents, ranks=None, wait=None):
    """Send one scheduler message to a worker set."""
    if ranks is None:
        ranks = range(len(bc_parents))
    # Each worker owns an independent shared-memory slot.  Do not wait for
    # rank 0 before publishing to rank 1: in TP/EP all workers must enter the
    # same NCCL collective, and rank 0 can otherwise block inside the first
    # all-reduce while rank 1 is still waiting for its control message.
    selected = [bc_parents[rank] for rank in ranks]
    for endpoint in selected:
        endpoint.send(msg, wait_ack=False)
    for endpoint in selected:
        endpoint.wait_ack(wait=wait)
