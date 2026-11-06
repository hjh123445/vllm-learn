"""_bucket_order must reproduce argsort(stable=True) bit-for-bit.

The EP dispatch and expert grouping rely on a stable grouping order; the
counting scatter is a kernel-count optimization (radix sort is multi-pass
and dominates short decode batches).  Pure CPU, no distributed setup.
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layers.moe import _bucket_order


def test_bucket_order_matches_stable_argsort():
    torch.manual_seed(777)
    for num_buckets in (1, 2, 8, 16, 64):
        for n in (0, 1, 7, 33, 1024, 16384):
            keys = torch.randint(0, num_buckets, (n,), dtype=torch.int64)
            reference = torch.argsort(keys, stable=True)
            counts = torch.bincount(keys, minlength=num_buckets)
            for got in (_bucket_order(keys, num_buckets),
                        _bucket_order(keys, num_buckets, counts)):
                assert got.dtype == reference.dtype
                assert torch.equal(got, reference), (num_buckets, n)


def test_bucket_order_accepts_int32_keys_and_empty_buckets():
    keys = torch.tensor([3, 3, 0, 3, 1], dtype=torch.int32)
    got = _bucket_order(keys, 6)
    reference = torch.argsort(keys.to(torch.int64), stable=True)
    assert torch.equal(got, reference)
    assert torch.equal(_bucket_order(torch.empty(0, dtype=torch.int64), 4),
                       torch.empty(0, dtype=torch.int64))
