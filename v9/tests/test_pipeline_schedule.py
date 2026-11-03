"""Pure scheduler-side tests for v9 PP microbatch plans."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.core import _spec_msg


class _Seq:
    def __init__(self, seq_id, tokens):
        self.seq_id = seq_id
        self.num_scheduled_tokens = tokens
        self.num_cached_tokens = 0
        self.num_prompt_tokens = tokens
        self.is_prefill = True
        self.block_table = [seq_id]
        self.temperature = 0.0
        self.last_token = 7
        self._tokens = list(range(tokens))

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, key):
        return self._tokens[key]


def _partition_by_count(seqs, count):
    count = max(1, min(count, len(seqs)))
    result = [[] for _ in range(count)]
    for i, seq in enumerate(seqs):
        result[min(i * count // len(seqs), count - 1)].append(seq)
    return [part for part in result if part]


def test_partition_preserves_order_and_ownership():
    seqs = [_Seq(i, n) for i, n in enumerate((4, 1, 3, 2))]
    chunks = _partition_by_count(seqs, 3)
    assert all(chunks)
    assert [s.seq_id for c in chunks for s in c] == [0, 1, 2, 3]
    assert sum(len(c) for c in chunks) == len(seqs)


def test_schedule_message_carries_plan_identity():
    msg = _spec_msg([_Seq(3, 2)], True, pipeline_lane=1, microbatch_id=5)
    assert msg["microbatch_id"] == 5
    assert msg["pipeline_lane"] == 1
    assert msg["is_prefill"] is True
    assert msg["scheduled"][0]["seq_id"] == 3
