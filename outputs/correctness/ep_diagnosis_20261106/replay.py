"""Route bijection and arithmetic replay, using only captured tiny tensors."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference', default='bf16_tp_v3')
parser.add_argument('--actual', default='bf16_ep')
args = parser.parse_args()
TP, EP = args.reference, args.actual


def load(mode, dp, tp, call):
    return torch.load(ROOT / mode / f'dp{dp}_tp{tp}' / f'call{call:03d}.pt', weights_only=True)


def weights(mode, dp, tp, layer):
    return torch.load(ROOT / mode / f'dp{dp}_tp{tp}' / f'weights_layer{layer}.pt', weights_only=True)


def error(a, b):
    d = a.double() - b.double()
    return dict(max_abs=d.abs().max().item() if d.numel() else 0,
                rmse=d.square().mean().sqrt().item() if d.numel() else 0,
                equal=torch.equal(a, b))


def expert(x, w13, w2, policy):
    if policy == 'float32':
        x, w13, w2 = x.float(), w13.float(), w2.float()
    if policy == 'float64':
        x, w13, w2 = x.double(), w13.double(), w2.double()
    gate, up = (x @ w13.t()).chunk(2, -1)
    act = F.silu(gate) * up
    if policy == 'down_float32':
        act, w2 = act.float(), w2.float()
    return act @ w2.t()


@torch.inference_mode()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report = dict(device=device, torch=torch.__version__, route_checks=[], local_replay=[])
    for call, layer in ((c, l) for c in range(4) for l in range(8)):
        label = f'model.layers.{layer}.mlp'
        sources = []
        for rank in range(4):
            dp, tp = divmod(rank, 2)
            rec = load(EP, dp, tp, call)
            x = rec['modules'][label]['input'][0]
            ids, ws = rec['modules'][label + '.router']['output']
            rows = (len(x) + 1) // 2
            x, ids, ws = x[tp * rows:(tp + 1) * rows], ids[tp * rows:(tp + 1) * rows], ws[tp * rows:(tp + 1) * rows]
            sources.append((x.repeat_interleave(2, 0), ids.flatten(), ws.flatten()))
        returned = [torch.empty_like(s[0]) for s in sources]
        total = 0
        for owner in range(4):
            rec = load(EP, owner // 2, owner % 2, call)
            xs, offsets, actual_raw = rec['experts'][str(layer)]
            selected = [torch.where(s[1] == owner)[0] for s in sources]
            expected = torch.cat([s[0][sel] for s, sel in zip(sources, selected)])
            assert torch.equal(xs, expected), (call, owner, 'dispatch routes')
            assert offsets.tolist() == [len(expected)]
            w = weights(EP, owner // 2, owner % 2, layer)
            assert w['e_start'] == owner
            raw = expert(xs.to(device), w['w13'][0].to(device), w['w2'][0].to(device), 'model').cpu()
            assert torch.equal(raw, actual_raw), (call, owner, 'expert replay')
            begin = 0
            for s, sel, out in zip(sources, selected, returned):
                out[sel] = actual_raw[begin:begin + len(sel)] * s[2][sel, None]
                begin += len(sel)
            total += len(expected)
        expected_routes = sum(len(s[1]) for s in sources)
        assert total == expected_routes
        for dp in range(2):
            out = torch.cat([returned[dp * 2 + tp].view(-1, 2, 16).float().sum(1).bfloat16()
                             for tp in range(2)])
            actual = load(EP, dp, 0, call)['modules'][label]['output']
            assert torch.equal(out, actual), (call, dp, 'combine order')
        report['route_checks'].append(dict(call=call, layer=layer, routes=total, dispatch_exact=True,
                                          expert_replay_exact=True, combine_exact=True))

        if layer != 0 or call not in (0, 3):
            continue
        for dp in range(2):
            a = load(TP, dp, 0, call)
            b = load(EP, dp, 0, call)
            x = a['modules'][label]['input'][0].to(device)
            assert torch.equal(x.cpu(), b['modules'][label]['input'][0])
            ids, ws = (v.to(device) for v in a['modules'][label + '.router']['output'])
            assert all(torch.equal(v.cpu(), z) for v, z in
                       zip((ids, ws), b['modules'][label + '.router']['output']))
            chunks = [weights(TP, dp, tp, 0) for tp in range(2)]
            full13 = torch.cat([torch.cat([w['w13'][:, :8] for w in chunks], 1),
                                torch.cat([w['w13'][:, 8:] for w in chunks], 1)], 1)
            full2 = torch.cat([w['w2'] for w in chunks], 2)
            for e in range(4):
                owner = weights(EP, e // 2, e % 2, 0)
                assert torch.equal(full13[e], owner['w13'][0])
                assert torch.equal(full2[e], owner['w2'][0])
            variants = {}
            for policy in ['model', 'down_float32', 'float32', 'float64']:
                dtype = {'model': torch.bfloat16, 'float64': torch.float64}.get(policy, torch.float32)
                routes = torch.empty((len(x), 2, 16), device=device, dtype=dtype)
                partials = [torch.empty_like(routes) for _ in range(2)]
                for e in range(4):
                    t, k = torch.where(ids == e)
                    inp = x[t]
                    val = expert(inp, full13[e].to(device), full2[e].to(device), policy)
                    routes[t, k] = val * ws[t, k, None].to(val.dtype)
                    for tp, w in enumerate(chunks):
                        val = expert(inp, w['w13'][e].to(device), w['w2'][e].to(device), policy)
                        partials[tp][t, k] = val * ws[t, k, None].to(val.dtype)
                # Production BF16 casts each TP partial before all-reduce.
                tp = sum((p.sum(1) if policy == 'float64' else
                          p.float().sum(1).to(dtype) for p in partials))
                full = routes.float().sum(1).to(dtype) if policy != 'float64' else routes.sum(1)
                variants[policy] = dict(tp_vs_full=error(tp, full),
                    full_vs_ep=error(full.cpu(), b['modules'][label]['output']),
                    tp_vs_baseline=error(tp.cpu(), a['modules'][label]['output']))
            report['local_replay'].append(dict(call=call, dp=dp, same_input=True,
                same_router=True, weight_reconstruction_exact=True, variants=variants))
    (ROOT / 'replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
