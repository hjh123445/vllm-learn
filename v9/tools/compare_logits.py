"""CPU-only fixed-prefix logit comparisons; never a throughput measurement."""
import argparse
import json
from pathlib import Path

import torch


def compare(actual, reference):
    a, b = actual.float(), reference.float()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Mismatched or nonfinite logits')
    av, ai = a.topk(min(10, a.numel()))
    bv, bi = b.topk(min(10, b.numel()))
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


def tensor_differences(actual, reference):
    """Compare cloned module boundaries, including BOTH router IDs/weights."""
    if isinstance(actual, torch.Tensor) and isinstance(reference, torch.Tensor):
        if actual.shape != reference.shape:
            return dict(shape_matches=False)
        delta = (actual.double() - reference.double()).abs()
        return dict(equal=torch.equal(actual, reference),
                    max_abs_error=float(delta.max()) if delta.numel() else 0.)
    if isinstance(actual, (tuple, list)) and isinstance(reference, (tuple, list)):
        if len(actual) != len(reference):
            raise ValueError('Module boundary arity changed')
        return [tensor_differences(a, b) for a, b in zip(actual, reference)]
    return dict(equal=actual == reference)


def compare_cached(reference_dir, actual_dir):
    """Align original cached calls; never compare already diverged prefixes.

    This intentionally consumes diagnose_ep's DP2/TP2 synthetic captures.
    TP0 carries replicated full logits; TP1 and weights remain in raw evidence.
    Intermediate chunk-prefill samples are reported but excluded from accepted
    generation counts. Module comparisons require the entire batch to match.
    """
    reference_dir, actual_dir = Path(reference_dir), Path(actual_dir)
    config = json.loads((reference_dir / 'results.json').read_text())
    other = json.loads((actual_dir / 'results.json').read_text())
    if config['prompts'] != other['prompts']:
        raise ValueError('Different original requests')
    def captures(root):
        active, idle = {}, []
        for path in sorted(root.glob('dp*_tp0/call*.pt')):
            record = torch.load(path, map_location='cpu', weights_only=True)
            key = str(path.relative_to(root))
            if record['sequences']:
                active[key] = record
            else:
                # EP-only idle participation has no sampler or query logits.
                idle.append(key)
        return active, idle
    refs, ref_idle = captures(reference_dir)
    acts, act_idle = captures(actual_dir)
    if not refs or refs.keys() != acts.keys():
        raise ValueError('Missing or extra active cached call captures')
    records, skipped, boundaries = [], [], []
    for path, ref in refs.items():
        act = acts[path]
        if (ref['dp_rank'], ref['index'], ref['is_prefill']) != (
                act['dp_rank'], act['index'], act['is_prefill']):
            raise ValueError('Call schedule changed; request alignment is required')
        tag = dict(dp=ref['dp_rank'], call=ref['index'], is_prefill=ref['is_prefill'])
        batch_equal = ref['sequences'] == act['sequences']
        if batch_equal:
            names = ['model.embed_tokens']
            layers = sorted({int(n.split('.')[2]) for n in ref['modules']
                             if n.startswith('model.layers.')})
            for layer in layers:
                names += [f'model.layers.{layer}.{suffix}' for suffix in
                          ('self_attn', 'mlp.router', 'mlp')]
            names += ['lm_head']
            boundaries.append(dict(**tag, outputs={name: tensor_differences(
                act['modules'][name]['output'], ref['modules'][name]['output'])
                for name in names}))
        ref_rows = {s['seq_id']: (i, s) for i, s in enumerate(ref['sequences'])}
        if set(ref_rows) != {s['seq_id'] for s in act['sequences']}:
            raise ValueError('Batch membership changed')
        for i, seq in enumerate(act['sequences']):
            j, original = ref_rows[seq['seq_id']]
            key = dict(**tag, seq=seq['seq_id'], prefix=seq['token_ids'])
            if any(seq[k] != original[k] for k in
                   ('token_ids', 'seq_len', 'cached', 'scheduled')):
                skipped.append(dict(**key, reason='different prefix or cache window'))
                continue
            a, b = act['modules']['sampler']['input'][0][i], ref['modules']['sampler']['input'][0][j]
            metrics = compare(a, b)
            prompt = config['prompts'][ref['dp_rank']::2][seq['seq_id']]
            position = len(seq['token_ids']) - len(prompt)
            candidates = sorted({metrics['actual_top1'], metrics['reference_top1']})
            records.append(dict(**key, generation_position=position,
                accepted_generation=position >= 0, batch_context_equal=batch_equal,
                actual_sample=int(act['modules']['sampler']['output'][i]),
                reference_sample=int(ref['modules']['sampler']['output'][j]),
                actual_sample_matches_argmax=int(act['modules']['sampler']['output'][i]) == metrics['actual_top1'],
                reference_sample_matches_argmax=int(ref['modules']['sampler']['output'][j]) == metrics['reference_top1'],
                candidate_logits={str(t): dict(actual=float(a[t]), reference=float(b[t])) for t in candidates},
                **metrics))
    accepted = [r for r in records if r['accepted_generation']]
    mismatches = sorted((r for r in accepted if not r['top1_matches']),
                        key=lambda r: (r['call'], r['dp'], r['seq']))
    return dict(scope='Original cached synthetic generation, TP0 replicated logits; not a benchmark',
        aligned_rows=len(records), accepted_rows=len(accepted), skipped_rows=len(skipped),
        accepted_top1_matches=sum(r['top1_matches'] for r in accepted),
        worst_max_abs_error=max(r['max_abs_error'] for r in records),
        worst_accepted_max_abs_error=max((r['max_abs_error'] for r in accepted), default=None),
        first_token_divergence=mismatches[0] if mismatches else None,
        generated_outputs_equal=config['outputs'] == other['outputs'],
        reference_outputs=config['outputs'], actual_outputs=other['outputs'],
        idle_collective_calls=dict(reference=ref_idle, actual=act_idle),
        records=records, skipped=skipped, module_boundaries=boundaries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root')
    ap.add_argument('--cached-reference', type=Path)
    ap.add_argument('--cached-actual', type=Path)
    ap.add_argument('--reference-dir', default='hf_reference')
    ap.add_argument('--pattern', default='teacher_*')
    ap.add_argument('--output', default='numerical_diagnostics.json')
    args = ap.parse_args()
    if args.cached_reference or args.cached_actual:
        if not (args.cached_reference and args.cached_actual) or args.root:
            ap.error('Use both --cached-reference and --cached-actual, without --root')
        result = compare_cached(args.cached_reference, args.cached_actual)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ('records', 'skipped', 'module_boundaries')}))
        return
    if not args.root:
        ap.error('--root is required for the historical fixed-prefix format')
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
