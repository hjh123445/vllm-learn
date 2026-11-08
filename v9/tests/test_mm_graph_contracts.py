"""Regression checks for complete-image chunking and static decode routing."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from engine.model_runner import ModelRunner
from engine.block_manager import BlockManager
from engine.sequence import Sequence
from engine.core import _spec_msg
from engine.vision_cache import VisionFeatureCache
from layers.moe import _local_route_weights
from models.qwen3_moe import Qwen3MoeForCausalLM


def runner():
    r = ModelRunner.__new__(ModelRunner)
    r.device = 'cpu'
    r.config = SimpleNamespace(mm_feature_cache_bytes=100000)
    r._mrope_rows = lambda seqs: None
    r.calls = []
    def encode(pixels, grid):
        assert pixels.shape[0] == int(grid.prod())
        r.calls.append(pixels.shape[0])
        # Depend on ALL patches so encoding a partial image cannot pass.
        n = pixels.shape[0] // 4
        merged = torch.arange(n * 4).view(n, 4).float() + pixels.sum()
        return merged, [merged + 100, merged + 200, merged + 300]
    r.model = SimpleNamespace(model=SimpleNamespace(vision_tower=object()), run_vision=encode)
    return r


def test_multimage_chunk_matches_full_and_encodes_once():
    mm = {'images': [dict(grid=[1, 4, 4], pos=2, pixels=np.ones((16, 24))),
                     dict(grid=[1, 4, 4], pos=9, pixels=np.ones((16, 24)) * 2)]}
    full = runner()
    s = SimpleNamespace(seq_id=1, num_cached_tokens=0, num_scheduled_tokens=15, mm=mm)
    expected = full._build_mm_batch([s])
    chunked = runner()
    actual, deep, positions = [], [[], [], []], []
    for start, end in ((0, 4), (4, 7), (7, 11), (11, 15)):
        s.num_cached_tokens, s.num_scheduled_tokens = start, end - start
        batch = chunked._build_mm_batch([s])
        if batch is None:
            continue
        actual.append(batch[0])
        for j, d in enumerate(batch[1]):
            deep[j].append(d)
        positions.append(batch[2] + start)
    torch.testing.assert_close(torch.cat(actual), expected[0])
    for j in range(3):
        torch.testing.assert_close(torch.cat(deep[j]), expected[1][j])
    assert torch.equal(torch.cat(positions), expected[2])
    assert chunked.calls == [16, 16]
    assert chunked._vision_features.bytes == 0


def test_feature_cache_is_bounded_and_eviction_is_safe():
    c = VisionFeatureCache(32)
    f = (torch.ones(2, 2), [torch.ones(2, 2)])
    c.put((1, 0), f); c.put((2, 0), f)
    assert c.bytes == 32 and c.get((1, 0)) is None
    assert c.get((2, 0)) is f
    c.clear()
    assert not c.entries and c.bytes == 0


def test_image_placeholders_never_share_text_apc():
    bm = BlockManager(16, 2, True)
    s = Sequence([1, 2, 3, 4]); s.block_size = 2
    bm.allocate(s, 0); s.num_scheduled_tokens = 4; bm.hash_blocks(s); bm.deallocate(s)
    image = Sequence([1, 2, 3, 4]); image.block_size = 2; image.mm = {'images': [object()]}
    assert bm.can_allocate(image) == 0
    bm.allocate(image, 0); image.num_scheduled_tokens = 4
    before = dict(bm.hash_to_block_id); bm.hash_blocks(image)
    assert bm.hash_to_block_id == before


def test_static_routing_handles_missing_local_experts():
    idx = torch.tensor([[0, 3], [1, 2], [3, 0]])
    weights = torch.tensor([[.25, .75], [.5, .5], [.8, .2]])
    actual = _local_route_weights(idx, weights, 2, 2)
    torch.testing.assert_close(actual, torch.tensor([[0., .5, 0.], [.75, 0., .8]]))
    assert actual.shape == (2, 3)


def test_logits_ignores_legacy_dump_environment(monkeypatch, tmp_path):
    model = Qwen3MoeForCausalLM.__new__(Qwen3MoeForCausalLM)
    nn.Module.__init__(model); model.lm_head = nn.Linear(4, 8)
    monkeypatch.setenv('VLLM_LOGIT_DUMP', str(tmp_path / 'forbidden'))
    x = torch.ones(2, 4)
    torch.testing.assert_close(model.compute_logits(x), model.lm_head(x))
    assert not (tmp_path / 'forbidden').exists()


def test_decode_message_excludes_raw_media():
    s = Sequence([1, 2]); s.num_scheduled_tokens = 1
    s.mm = {'images': [object()], 'pos3': object(), 'delta': -4}
    assert _spec_msg([s], False)['scheduled'][0]['mm'] == {'delta': -4}


def test_graph_predicate_distinguishes_attention_and_expert_backends(monkeypatch):
    from layers.parallel import configure_parallel
    from layers.attention import Attention
    from layers.moe import MoEMLP
    from layers import kernels
    configure_parallel(tp_size=2, enable_expert_parallel=True)
    monkeypatch.setenv('VLLM_LEARN_DECODE_GRAPH', '1')
    monkeypatch.setenv('VLLM_LEARN_EP_DENSE_DECODE', '1')
    monkeypatch.setenv('VLLM_LEARN_DENSE_EXPERT_BACKEND', 'bmm')
    monkeypatch.setenv('VLLM_LEARN_EXPERT_BACKEND', 'torch_grouped')
    monkeypatch.setenv('VLLM_LEARN_ATTENTION_BACKEND', 'triton')
    monkeypatch.setattr(kernels, '_REFERENCE_QK', False)
    r = ModelRunner.__new__(ModelRunner)
    r.config = SimpleNamespace(batch_execution_mode='batched')
    try:
        r.model = nn.ModuleList([Attention(1, 16, .25), MoEMLP(SimpleNamespace(
            hidden_size=16, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16))])
        assert r._decode_graph_supported() == (True, '')
        r.model[0].backend = 'sdpa_reference'
        assert not r._decode_graph_supported()[0]
        with pytest.raises(ValueError, match='requested but unsupported'):
            r._setup_decode_graphs()
        r.decode_graphs = {1: {}, 4: {}}
        monkeypatch.setenv('VLLM_LEARN_TP_REDUCTION_DTYPE', 'model')
        assert r._pick_decode_graph(3) == 4
        assert r._pick_decode_graph(4) == 4
        monkeypatch.setenv('VLLM_LEARN_TP_REDUCTION_DTYPE', 'float32')
        assert r._pick_decode_graph(3) == 4
        assert r._pick_decode_graph(5) is None
    finally:
        configure_parallel()


def test_sparse_graph_admission_and_invalid_parallelism(monkeypatch):
    from layers.parallel import configure_parallel
    from layers.moe import MoEMLP
    from layers import kernels
    monkeypatch.setenv('VLLM_LEARN_DECODE_GRAPH', '1')
    monkeypatch.setenv('VLLM_LEARN_EP_DENSE_DECODE', '0')
    monkeypatch.delenv('VLLM_LEARN_EP_SPARSE_DECODE', raising=False)
    monkeypatch.setattr(kernels, '_REFERENCE_QK', False)
    cfg = SimpleNamespace(hidden_size=16, num_experts=8,
                          num_experts_per_tok=2, moe_intermediate_size=16)
    configure_parallel(tp_size=2, enable_expert_parallel=True)
    try:
        m = MoEMLP(cfg)
        assert m.ep_sparse_decode and not m.ep_dense_decode
        r = ModelRunner.__new__(ModelRunner)
        r.config = SimpleNamespace(batch_execution_mode='batched')
        r.model = nn.ModuleList([m])
        assert r._decode_graph_supported() == (True, '')
        m.ep_sparse_decode = False
        assert not r._decode_graph_supported()[0]
        monkeypatch.setenv('VLLM_LEARN_EP_SPARSE_DECODE', '1')
        monkeypatch.setenv('VLLM_LEARN_EP_DENSE_DECODE', '1')
        with pytest.raises(ValueError, match='not both'):
            MoEMLP(cfg)
        monkeypatch.setenv('VLLM_LEARN_EP_DENSE_DECODE', '0')
        configure_parallel(tp_size=2, dp_size=2, enable_expert_parallel=True)
        with pytest.raises(ValueError, match='DP=1'):
            MoEMLP(cfg)
    finally:
        configure_parallel()
