"""Real CPU profiler verifies range nesting, teardown and exceptional forwards."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.module_profiling import ModuleProfileRanges


class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(4, 4)

    def forward(self, x):
        return self.router(x).relu()


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = ToyMoE()

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity(), ToyBlock()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_ranges_preserve_output_and_remove_hooks():
    model = ToyModel()
    x = torch.randn(3, 4)
    expected = model(x)
    ranges = ModuleProfileRanges(model, stage=2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with ranges:
            ranges.install()  # No duplicate registration.
            actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    events = {event.name: event for event in prof.events() if event.name.startswith("stage")}
    assert set(events) == {"stage2.layer1", "stage2.layer1.attention",
                           "stage2.layer1.moe", "stage2.layer1.moe.router"}
    assert events["stage2.layer1.attention"].cpu_parent is events["stage2.layer1"]
    assert events["stage2.layer1.moe.router"].cpu_parent is events["stage2.layer1.moe"]
    assert any(event.name == "aten::linear" for event in prof.events())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as plain:
        model(x)
    assert not any(event.name.startswith("stage") for event in plain.events())
    # A collector can be reused for a later profiling window.
    with ranges:
        model(x)
    assert all(not m._forward_hooks for m in model.modules())


def test_exception_closes_nested_ranges_and_removes_hooks():
    model = ToyModel()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with pytest.raises(RuntimeError):
            with ModuleProfileRanges(model):
                model(torch.randn(1, 5))  # Linear input mismatch inside Attention.
        with torch.profiler.record_function("after_failure"):
            torch.ones(1)
    events = {event.name: event for event in prof.events()}
    assert events["after_failure"].cpu_parent is None
    assert events["stage0.layer1.attention"].cpu_parent is events["stage0.layer1"]
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_removing_profile_ranges_preserves_other_observers():
    model = ToyModel()
    calls = []
    handle = model.layers[1].register_forward_hook(lambda *_: calls.append(1))
    try:
        with ModuleProfileRanges(model):
            model(torch.randn(1, 4))
        model(torch.randn(1, 4))
        assert calls == [1, 1]
        assert len(model.layers[1]._forward_hooks) == 1
    finally:
        handle.remove()
