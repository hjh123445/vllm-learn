"""Minimal safetensors reader over the HF checkpoint: returns memory-mapped views so 189 GiB Engram
tables and 300 GiB of experts can be sliced without a torch dtype for F8_E8M0 or a full copy."""
import json
import os
import struct

import numpy as np
import torch

DTYPE_BYTES = {"F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}
TORCH_DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.uint8,  # kept as raw bytes; see quant.e8m0_to_float
    "I8": torch.int8,
    "U8": torch.uint8,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I32": torch.int32,
    "I64": torch.int64,
}


class Checkpoint:
    def __init__(self, path: str):
        self.path = path
        self.weight_map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self._mmaps: dict[str, np.memmap] = {}

    def _header(self, fn: str):
        if fn not in self._headers:
            with open(os.path.join(self.path, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[fn] = (json.loads(f.read(n)), 8 + n)
        return self._headers[fn]

    def _mmap(self, fn: str) -> np.memmap:
        if fn not in self._mmaps:
            self._mmaps[fn] = np.memmap(os.path.join(self.path, fn), dtype=np.uint8, mode="r")
        return self._mmaps[fn]

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def names(self, prefix: str = "") -> list[str]:
        return [n for n in self.weight_map if n.startswith(prefix)]

    def meta(self, name: str) -> tuple[str, list[int]]:
        hdr, _ = self._header(self.weight_map[name])
        m = hdr[name]
        return m["dtype"], m["shape"]

    def get(self, name: str, device=None, rows: slice | None = None) -> torch.Tensor:
        """Tensor for `name` (a copy on `device`, or a CPU tensor over the mmap when device is None).
        `rows` slices the leading dimension without touching the rest of the file."""
        fn = self.weight_map[name]
        hdr, base = self._header(fn)
        m = hdr[name]
        dtype, shape = m["dtype"], list(m["shape"])
        start, end = m["data_offsets"]
        if rows is not None:
            row_bytes = int(np.prod(shape[1:])) * DTYPE_BYTES[dtype] if len(shape) > 1 else DTYPE_BYTES[dtype]
            r0, r1, _ = rows.indices(shape[0])
            start, end = start + r0 * row_bytes, start + r1 * row_bytes
            shape[0] = r1 - r0
        buf = self._mmap(fn)[base + start : base + end]
        t = torch.frombuffer(memoryview(buf), dtype=torch.uint8).view(TORCH_DTYPE[dtype]).view(shape)
        return t if device is None else t.to(device, non_blocking=False)
