"""Offline analysis must distinguish cache divergence, ties and idle EP calls."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compare_logits import compare, compare_cached


def capture(root, prefixes, rows, idle=False):
    rank = root / 'dp0_tp0'
    rank.mkdir(parents=True)
    (root / 'results.json').write_text(json.dumps(dict(prompts=[[1, 2]], outputs=[])))
    for i, (prefix, row) in enumerate(zip(prefixes, rows)):
        logits = torch.tensor([row])
        torch.save(dict(dp_rank=0, index=i, is_prefill=False,
            sequences=[dict(seq_id=0, token_ids=prefix, seq_len=len(prefix),
                            cached=len(prefix)-1, scheduled=1)],
            modules={'model.embed_tokens': dict(output=torch.ones(1, 2)),
                     'lm_head': dict(output=logits),
                     'sampler': dict(input=(logits,), output=logits.argmax(-1))}),
                   rank / f'call{i:03d}.pt')
    if idle:
        torch.save(dict(sequences=[]), rank / 'call099.pt')


def test_ties_use_argmax_and_small_vocab():
    result = compare(torch.tensor([1., 1., 0.]), torch.tensor([.9, 1., 0.]))
    assert result['actual_top1'] == 0 and result['reference_top1'] == 1
    assert result['actual_tied_winners'] == 2
    assert result['reference_winner_is_actual_max']


def test_cached_comparison_excludes_changed_prefixes_and_idle_calls(tmp_path):
    capture(tmp_path / 'ref', [[1], [1, 2], [1, 2, 1]],
            [[0., 1., 0.]] * 3)
    capture(tmp_path / 'act', [[1], [1, 2], [1, 2, 2]],
            [[0., 1., 0.], [0., .9, 1.], [1., 0., 0.]], idle=True)
    result = compare_cached(tmp_path / 'ref', tmp_path / 'act')
    assert result['aligned_rows'] == 2 and result['accepted_rows'] == 1
    assert result['skipped_rows'] == 1 and result['accepted_top1_matches'] == 0
    assert result['first_token_divergence']['call'] == 1
    assert len(result['idle_collective_calls']['actual']) == 1
