"""Shared-memory control-plane channels for EngineCore workers.

The payload is serialized once into a fixed shared-memory slot. A small
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

    def _buffer(self):
        if self._shm is None:
            self._shm = SharedMemory(name=self.shm_name)
        return self._shm.buf

    def send(self, message):
        if not self.sender:
            raise RuntimeError("shared-memory endpoint is receive-only")
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > self.capacity:
            raise ValueError(
                f"scheduler message is {len(payload)} bytes; shared-memory "
                f"slot capacity is {self.capacity} bytes")
        buf = self._buffer()
        buf[:len(payload)] = payload
        self.notify.send(len(payload))
        # One producer/consumer pair owns the slot. This acknowledgement is
        # what makes reuse safe without a heavyweight lock.
        self.ack.recv()

    def recv(self):
        if self.sender:
            raise RuntimeError("shared-memory endpoint is send-only")
        size = self.notify.recv()
        message = pickle.loads(bytes(self._buffer()[:size]))
        self.ack.send(True)
        return message

    def poll(self, timeout=0.0):
        if self.sender:
            raise RuntimeError("shared-memory endpoint is send-only")
        return self.notify.poll(timeout)

    def close(self):
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


def _channel_pair(reverse=False):
    # notify: producer -> consumer; ack: consumer -> producer
    notify_recv, notify_send = mp.Pipe(duplex=False)
    ack_recv, ack_send = mp.Pipe(duplex=False)
    shm = SharedMemory(create=True, size=_DEFAULT_CAPACITY)
    if reverse:
        # Parent receives, spawned child sends.
        parent = SharedMemoryEndpoint(shm.name, notify_recv, ack_send,
                                      owner=True, sender=False)
        child = SharedMemoryEndpoint(shm.name, notify_send, ack_recv,
                                     owner=False, sender=True)
        parent._shm = shm
        return parent, child
    parent = SharedMemoryEndpoint(shm.name, notify_send, ack_recv,
                                  owner=True, sender=True)
    child = SharedMemoryEndpoint(shm.name, notify_recv, ack_send,
                                 owner=False, sender=False)
    parent._shm = shm
    return parent, child


def make_channels(world_size: int):
    """Return shared-memory broadcast and response endpoint pairs."""
    bc = [_channel_pair() for _ in range(world_size)]
    rc = [_channel_pair(reverse=True) for _ in range(world_size)]
    return bc, rc


def broadcast(msg, bc_parents, ranks=None):
    """Send one scheduler message to a worker set."""
    if ranks is None:
        ranks = range(len(bc_parents))
    for rank in ranks:
        bc_parents[rank].send(msg)
