"""Explicit computation segments around dynamic distributed/host boundaries.

No collectives are captured here. Every topology keeps its existing transport
protocol, including idle-DP participation and PP credits. Segments use exact
input shapes; new shapes capture lazily. A bounded cache fails explicitly on
budget exhaustion, never silently labels eager work as graph replay.
"""
from contextlib import contextmanager
import contextvars
from collections import OrderedDict

import torch

_ACTIVE = contextvars.ContextVar('piecewise_decode_graph', default=None)


def execute(tag, function, inputs, signature=()):
    manager = _ACTIVE.get()
    if manager is None or any(not x.is_cuda or not x.numel() for x in inputs):
        return function(*inputs)
    return manager.execute(tag, function, inputs, signature)


class PiecewiseGraphs:
    def __init__(self, budget):
        self.budget = budget
        self.entries = OrderedDict()
        self.bytes = 0
        self.captures = 0
        self.replays = 0

    @contextmanager
    def active(self):
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def execute(self, tag, function, inputs, signature):
        key = (tag, signature, tuple((tuple(x.shape), x.dtype, x.device) for x in inputs))
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
        if entry is None:
            # Disable nested segmentation while tracing this compute segment.
            token = _ACTIVE.set(None)
            graph = None
            try:
                before = torch.cuda.memory_allocated()
                reserved_before = torch.cuda.memory_reserved()
                buffers = tuple(x.detach().clone() for x in inputs)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        result = function(*buffers)
                torch.cuda.current_stream().wait_stream(stream)
                del result
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                # Private pools: arbitrary shapes may replay in arbitrary order.
                with torch.cuda.graph(graph):
                    output = function(*buffers)
                cost = max(2**21, torch.cuda.memory_allocated() - before,
                           torch.cuda.memory_reserved() - reserved_before)
                if cost > self.budget:
                    graph.reset()
                    raise RuntimeError('piecewise CUDA Graph cache exceeds decode_graph_reserve_bytes; '
                                       'increase reserve or reduce workload shapes')
                while self.entries and self.bytes + cost > self.budget:
                    torch.cuda.synchronize()
                    _, old = self.entries.popitem(last=False)
                    old['graph'].reset()
                    self.bytes -= old['bytes']
                    del old
                # The closure owns any external weights/views used by capture;
                # CUDA graph pointers alone do not keep their storage alive.
                entry = dict(graph=graph, inputs=buffers, output=output, bytes=cost,
                             function=function)
                self.entries[key] = entry
                self.bytes += cost
                self.captures += 1
            finally:
                _ACTIVE.reset(token)
        for buffer, value in zip(entry['inputs'], inputs):
            buffer.copy_(value)
        entry['graph'].replay()
        self.replays += 1
        # Consumers may retain outputs while the same segment is replayed for
        # another microbatch; copy them out of the persistent graph buffers.
        def copy(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, tuple):
                return tuple(copy(v) for v in value)
            raise TypeError('piecewise graph output must be tensors or tuples')
        return copy(entry['output'])

    def close(self):
        for entry in self.entries.values():
            entry['graph'].reset()
        self.entries.clear()
        self.bytes = 0
