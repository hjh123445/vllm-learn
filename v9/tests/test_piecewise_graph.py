"""Actual changed-content replay, graph output lifetime and bounded-cache behavior."""
import pytest
import torch
from utils.piecewise_graph import PiecewiseGraphs, execute


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_segments_replay_new_values_and_keep_returned_outputs():
    manager = PiecewiseGraphs(16 * 2**20)
    x = torch.arange(24, device='cuda', dtype=torch.float32).view(3, 8)
    fn = lambda value: (value.square() + 2, value.sum(-1))
    try:
        with manager.active():
            first = execute('test', fn, (x,))
            x.add_(3)
            second = execute('test', fn, (x,))
        assert manager.captures == 1 and manager.replays == 2
        assert torch.equal(second[0], x.square() + 2)
        assert torch.equal(second[1], x.sum(-1))
        assert torch.equal(first[0], (x-3).square() + 2)
    finally:
        manager.close()
    assert not manager.entries


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_segments_evict_old_shapes_without_eager_fallback():
    manager = PiecewiseGraphs(64 * 2**20)
    try:
        for n in (3, 7, 3):
            x = torch.arange(n, device='cuda', dtype=torch.float32)
            with manager.active():
                out = execute('add', lambda value: value + 7, (x,))
            assert torch.equal(out, x + 7)
            manager.budget = manager.bytes
        assert manager.captures == manager.replays == 3
        assert len(manager.entries) == 1
        assert manager.bytes <= manager.budget
    finally:
        manager.close()
