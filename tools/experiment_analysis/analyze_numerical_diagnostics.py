"""CPU-only fixed-prefix logit comparisons; never a throughput measurement."""
import argparse
import json
from pathlib import Path

import torch


def compare(actual, reference):
    a, b = actual.float(), reference.float()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Mismatched or nonfinite logits')
    av, ai = a.topk(10)
    bv, bi = b.topk(10)
    # Greedy sampling uses argmax (first maximal index), whereas topk has no
    # stable tie order. Report ties explicitly instead of inventing divergence.
    actual_winner, reference_winner = int(a.argmax()), int(b.argmax())
    delta = (a-b).abs()
    margin = float(bv[0]-bv[1])
    return dict(actual_top1=actual_winner, reference_top1=reference_winner,
        top1_matches=actual_winner == reference_winner, reference_top1_margin=margin,
        actual_tied_winners=int((a == a.max()).sum()),
        reference_tied_winners=int((b == b.max()).sum()),
        actual_winner_is_reference_max=bool(b[actual_winner] == b.max()),
        reference_winner_is_actual_max=bool(a[reference_winner] == a.max()),
        topk_tie_order_changes_actual_winner=int(ai[0]) != actual_winner,
        topk_tie_order_changes_reference_winner=int(bi[0]) != reference_winner,
        actual_top1_margin=float(av[0]-av[1]), max_abs_error=float(delta.max()),
        mean_abs_error=float(delta.mean()), rmse=float((a-b).square().mean().sqrt()),
        centered_cosine=float(torch.nn.functional.cosine_similarity(
            a-a.mean(), b-b.mean(), dim=0)),
        probability_total_variation=float((a.softmax(0)-b.softmax(0)).abs().sum()/2),
        reference_winner_error=float(delta[reference_winner]),
        top1_stability_bound_satisfied=bool(2*delta.max() < margin),
        top10_overlap=len(set(ai.tolist()) & set(bi.tolist())),
        actual_top10=ai.tolist(), reference_top10=bi.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--reference-dir', default='hf_reference')
    ap.add_argument('--pattern', default='teacher_*')
    ap.add_argument('--output', default='numerical_diagnostics.json')
    args = ap.parse_args()
    root = Path(args.root)
    hf_dir = root / args.reference_dir
    hf = json.loads((hf_dir/'text_reference.json').read_text())
    refs = {}
    for item in hf:
        logits = torch.load(hf_dir/'logits'/Path(item['raw_logits_file']).name,
                            map_location='cpu', weights_only=True)
        for i, row in enumerate(logits):
            refs[(item['workload'], item['index'], i)] = row
    summary = {}
    captured = {}
    for folder in sorted(root.glob(args.pattern)):
        if not (folder/'requests.json').exists():
            continue
        requests = json.loads((folder/'requests.json').read_text())
        by_seq = {}
        for path in sorted((folder/'logits').glob('*.pt')):
            batch = torch.load(path, map_location='cpu', weights_only=True)
            for seq, row in zip(batch['seq_ids'], batch['logits']):
                if seq in by_seq:
                    raise ValueError(f'{folder.name}: duplicate terminal capture {seq}')
                by_seq[seq] = row
        records = []
        captured[folder.name] = {}
        for request in requests:
            key = (request['workload'], request['index'], request['prefix_tokens'])
            actual = by_seq[request['seq_id']]
            captured[folder.name][key] = actual
            metrics = compare(actual, refs[key])
            records.append(dict(workload=key[0], index=key[1], prefix_tokens=key[2],
                expected_generated_token=request['expected_next_token'],
                generated_matches_raw_hf_argmax=(request['expected_next_token'] == metrics['reference_top1']),
                sampled_matches_captured_argmax=(request['actual_next_token'] == metrics['actual_top1']),
                **metrics))
        summary[folder.name] = dict(cases=len(records),
            raw_top1_matches=sum(r['top1_matches'] for r in records),
            worst_max_abs_error=max(r['max_abs_error'] for r in records), records=records)
    cross = {}
    names = sorted(captured)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            keys = sorted(captured[left].keys() & captured[right].keys())
            rows = [dict(workload=k[0], index=k[1], prefix_tokens=k[2],
                         **compare(captured[right][k], captured[left][k])) for k in keys]
            cross[left+'__'+right] = dict(cases=len(rows),
                raw_top1_matches=sum(r['top1_matches'] for r in rows), records=rows)
    result = dict(scope='Fixed HF continuation prefixes, terminal query logits; CPU analysis',
        caveat='Greedy winners use torch.argmax; topk is only used for margins and overlap. '
               'Tied maxima are reported separately. No blanket BF16 attribution. '
               'Raw HF logits compared separately from generate processors. '
               'The 2*Linf < margin bound is sufficient, not necessary for top1 stability.',
        against_hf=summary, cross_implementation=cross)
    (root/args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {a:b for a,b in v.items() if a != 'records'} for k,v in summary.items()}))


if __name__ == '__main__':
    main()
