"""Exercise startup and replay contracts without allocating a CUDA context."""
from types import SimpleNamespace
import os
import pytest
import torch
from engine.model_runner import ModelRunner
from engine.sequence import Sequence
from layers.parallel import configure_parallel


class FakeModel:
    def eval(self):
        return self

    def __call__(self, ids, positions, **kwargs):
        return ids.float().unsqueeze(-1)

    def compute_logits(self, hidden):
        return hidden


def test_requested_graph_allows_initialization_warmup_only(monkeypatch):
    import engine.model_runner as module
    configure_parallel()
    monkeypatch.setenv('VLLM_LEARN_DECODE_GRAPH', '1')
    monkeypatch.delenv('VLLM_LEARN_DECODE_GRAPH_STRICT', raising=False)
    monkeypatch.setenv('VLLM_LEARN_BATCH_INVARIANT', '0')
    monkeypatch.delenv('VLLM_LEARN_MM', raising=False)
    monkeypatch.setattr(module, 'load_thinker_config', lambda _: {})
    monkeypatch.setattr(module, 'Qwen3MoeForCausalLM', lambda *args: FakeModel())
    monkeypatch.setattr(module, 'load_model', lambda *args, **kwargs: None)
    monkeypatch.setattr(ModelRunner, 'allocate_kv_cache', lambda self: None)
    monkeypatch.setattr(ModelRunner, '_setup_decode_graphs', lambda self: None)
    def warmup(self):
        assert self._warming_up
        assert os.environ['VLLM_LEARN_BATCH_INVARIANT'] == '1'
        self.device = 'cpu'
        self.prepare_decode = lambda seqs: (torch.tensor([7]), torch.tensor([0]))
        self.sampler = lambda logits, temperatures: logits[:, 0].long()
        assert self.run([SimpleNamespace(last_token=7, temperature=0.)], False) == [7]
    monkeypatch.setattr(ModelRunner, '_warmup_model', warmup)
    runner = ModelRunner(SimpleNamespace(model='unused', kvcache_block_size=256,
                                         max_model_len=1024, batch_execution_mode='batched'))
    assert not runner._warming_up and runner._decode_graph_strict
    with pytest.raises(RuntimeError, match='no captured CUDA Graph bucket'):
        runner.run([SimpleNamespace(last_token=7, temperature=0.)], False)


def replay_runner():
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 256
    runner.config = SimpleNamespace(max_model_len=1024)
    runner._graph_dummy_block = 16
    runner.graph_replay_count = 0
    calls = []
    state = dict(ids=torch.zeros(4, dtype=torch.int64), pos=torch.zeros(4, dtype=torch.int64),
                 slots=torch.zeros(4, dtype=torch.int32), lens=torch.zeros(4, dtype=torch.int32),
                 bt=torch.zeros(4, 4, dtype=torch.int32), logits=torch.zeros(4, 8),
                 graph=SimpleNamespace(replay=lambda: calls.append(1)))
    return runner, state, calls


def test_replay_block_boundary_padding_and_empty():
    runner, state, calls = replay_runner()
    seq = Sequence([9] * 257)
    seq.block_table = [5, 7]
    runner._replay_decode_graph(state, [seq])
    assert state['slots'].tolist() == [7 * 256, -1, -1, -1]
    assert state['lens'].tolist() == [257, 1, 1, 1]
    assert state['bt'][0].tolist() == [5, 7, -1, -1]
    assert state['bt'][1:].eq(16).all()
    assert calls == [1] and runner.graph_replay_count == 1
    assert runner._replay_decode_graph(state, []).shape == (0, 8)
    assert calls == [1]


@pytest.mark.parametrize('length,table', [(257, [5]), (12, [16]), (12, [-1]),
                                         (12, [1, 2, 3, 4, 5]), (1025, [1, 2, 3, 4, 5])])
def test_invalid_kv_metadata_never_launches(length, table):
    runner, state, calls = replay_runner()
    seq = Sequence([1] * length)
    seq.block_table = table
    before = state['ids'].clone()
    with pytest.raises(ValueError):
        runner._replay_decode_graph(state, [seq])
    assert not calls and torch.equal(before, state['ids'])


def test_oversized_batch_never_launches():
    runner, state, calls = replay_runner()
    with pytest.raises(ValueError, match='exceeds captured bucket'):
        runner._replay_decode_graph(state, [Sequence([1])] * 5)
    assert not calls


def test_mixed_admission_preserves_decode_path_and_output_order(monkeypatch):
    configure_parallel()
    monkeypatch.setenv('VLLM_LEARN_BATCH_INVARIANT', '1')
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(batch_execution_mode='batched')
    seqs = [SimpleNamespace(num_cached_tokens=cached, num_prompt_tokens=20, output=output)
            for cached, output in [(0, 11), (25, 22), (7, 33), (20, 44)]]
    calls = []
    def forward(subset, is_prefill, microbatch_id):
        calls.append(([seq.output for seq in subset], is_prefill, microbatch_id))
        return [seq.output for seq in subset]
    runner.run = forward
    assert ModelRunner.run(runner, seqs, True, microbatch_id=9) == [11, 22, 33, 44]
    assert calls == [([22, 44], False, 9), ([11, 33], True, 9)]
