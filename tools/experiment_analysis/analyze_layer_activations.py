"""CPU-only activation comparison, matching tensor roles rather than tuples."""
import argparse
import json
from pathlib import Path
import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensors(v)]
    return []


def metrics(a, b):
    if a.shape != b.shape:
        return dict(shape_a=list(a.shape), shape_b=list(b.shape), comparable=False)
    af, bf = a.float().flatten(), b.float().flatten()
    delta = af - bf
    return dict(comparable=True, shape=list(a.shape), dtype_a=str(a.dtype), dtype_b=str(b.dtype),
                max_abs=delta.abs().max().item() if delta.numel() else 0.,
                rmse=delta.square().mean().sqrt().item() if delta.numel() else 0.,
                reference_rms=af.square().mean().sqrt().item() if af.numel() else 0.,
                equal_fraction=(a == b).float().mean().item() if af.numel() else 1.,
                cosine=torch.nn.functional.cosine_similarity(af[None], bf[None]).item() if af.numel() else 1.)


def compare(reference, candidate):
    rows = []
    for path in sorted(reference.glob('stage*.pt')):
        other = candidate/path.name
        if not other.exists():
            rows.append(dict(file=path.name, missing=True))
            continue
        a, b = torch.load(path, weights_only=True), torch.load(other, weights_only=True)
        row = dict(file=path.name, label=a['label'])
        for field in ('input', 'output'):
            aa, bb = tensors(a[field]), tensors(b[field])
            if row['label'].endswith('.router') and field == 'output':
                # HF router may return (logits, weights, indices), custom (indices, weights).
                ai = next((t for t in aa if not t.is_floating_point()), None)
                bi = next((t for t in bb if not t.is_floating_point()), None)
                if ai is not None and bi is not None and ai.shape == bi.shape:
                    row['router_index_equal_fraction'] = (ai == bi).float().mean().item()
                    aset, bset = ai.sort(-1).values, bi.sort(-1).values
                    row['router_set_equal_rows'] = (aset == bset).all(-1).float().mean().item()
                row['router_output_shapes'] = [[list(t.shape), str(t.dtype)] for t in aa]
                row['custom_router_output_shapes'] = [[list(t.shape), str(t.dtype)] for t in bb]
                continue
            if aa and bb:
                row[field] = metrics(aa[0], bb[0])
        rows.append(row)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    torch.set_num_threads(4)
    result = {}
    for name in ('layer_custom', 'layer_custom_reference_math',
                 'layer_custom_sdpa', 'layer_custom_sdpa_hf_reduce'):
        if not (root/name/'metadata.json').exists():
            continue
        rows = compare(root/'layer_hf', root/name)
        result[name] = rows
        print(name, 'rows', len(rows))
        for row in rows:
            if row.get('label', '').startswith('layer0.') or row.get('label') == 'embedding':
                print(json.dumps(row))
    (root/'layer_differences.json').write_text(json.dumps(result, indent=2))
