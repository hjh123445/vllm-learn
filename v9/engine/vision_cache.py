"""Bounded, request-scoped cache of complete image encoder outputs.

LLM prefill may split visual tokens; the bidirectional vision encoder may
not split raw patches. Cache merged/deepstack features, then slice those.
No cross-request reuse until media identity participates in cache keys.
"""
from collections import OrderedDict


class VisionFeatureCache:
    def __init__(self, max_bytes):
        if max_bytes < 0:
            raise ValueError('mm_feature_cache_bytes must be nonnegative')
        self.max_bytes = max_bytes
        self.bytes = 0
        self.entries = OrderedDict()

    def pop(self, key):
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.bytes -= entry[1]
            return entry[0]
        return None

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            return None
        self.entries.move_to_end(key)
        return entry[0]

    def put(self, key, features):
        self.pop(key)
        merged, deepstack = features
        size = sum(t.numel() * t.element_size() for t in (merged, *deepstack))
        if size > self.max_bytes:
            return
        while self.entries and self.bytes + size > self.max_bytes:
            _, (_, old_size) = self.entries.popitem(last=False)
            self.bytes -= old_size
        self.entries[key] = (features, size)
        self.bytes += size

    def clear(self):
        self.entries.clear()
        self.bytes = 0
