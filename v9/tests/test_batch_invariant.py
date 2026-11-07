"""Fixed-order math must match a numeric oracle and survive padding/reorder."""
import pytest
import torch
from layers.batch_invariant import linear, expert_sum
from layers.kernels import fused_add_rmsnorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('experts', [None, 3])
def test_fixed_linear_padding_and_permutation(monkeypatch, experts):
    monkeypatch.setenv('VLLM_LEARN_BATCH_INVARIANT', '1')
    torch.manual_seed(193)
    shape = (33, 96) if experts is None else (experts, 33, 96)
    weight_shape = (80, 96) if experts is None else (experts, 80, 96)
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(weight_shape, device='cuda', dtype=x.dtype)
    full = linear(x, weight)
    oracle = x.float() @ weight.float().transpose(-1, -2)
    torch.testing.assert_close(full.float(), oracle, atol=.125, rtol=.01)
    for n in (1, 3, 7, 8, 17):
        actual = linear(x[..., :n, :], weight)
        assert torch.equal(actual, full[..., :n, :])
    order = torch.randperm(33, device='cuda')
    assert torch.equal(linear(x[..., order, :], weight), full[..., order, :])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_fixed_norm_and_expert_sum_padding(monkeypatch):
    monkeypatch.setenv('VLLM_LEARN_BATCH_INVARIANT', '1')
    torch.manual_seed(619)
    x = torch.randn((8, 2048), device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(2048, device='cuda', dtype=x.dtype)
    full, _ = fused_add_rmsnorm(x, residual.clone(), weight, 1e-6)
    for n in (1, 3, 5, 7):
        actual, _ = fused_add_rmsnorm(x[:n], residual[:n].clone(), weight, 1e-6)
        assert torch.equal(actual, full[:n])
    contributions = torch.randn((16, 8, 2048), device='cuda', dtype=x.dtype)
    reduced = expert_sum(contributions)
    oracle = torch.zeros_like(reduced)
    for expert in contributions: oracle += expert.float()
    assert torch.equal(reduced, oracle)
    assert torch.equal(expert_sum(contributions[:, :7]), reduced[:7])
