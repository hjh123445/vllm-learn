"""CUDA expert-core checks: sparse vs strict dense, graph route changes/padding.

Synthetic weights/inputs, no model-generation or throughput claims. Full-model
coverage belongs to validate_decode_graph.py --compare-dense.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['VLLM_LEARN_BATCH_INVARIANT'] = '1'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    import torch
    from layers.sparse_decode import sparse_expert_routes
    from layers.batch_invariant import linear, expert_sum
    from layers.kernels import silu_and_mul
    from layers.moe import _local_route_weights
    torch.cuda.set_device(0)
    torch.manual_seed(97)
    rows = []
    for dtype in (torch.bfloat16, torch.float16):
        for h, inter, local_e, first, global_e, k in ((80, 48, 3, 2, 8, 3),
                                                       (2048, 768, 16, 32, 128, 8)):
            w13 = torch.randn(local_e, 2 * inter, h, device='cuda', dtype=dtype) * .02
            w2 = torch.randn(local_e, h, inter, device='cuda', dtype=dtype) * .02
            for t in (1, 3, 8, 17, 32, 65):
                x = torch.randn(t, h, device='cuda', dtype=dtype)
                idx = torch.empty((t, k), device='cuda', dtype=torch.int64)
                weights = torch.empty((t, k), device='cuda', dtype=dtype)
                def sparse():
                    return sparse_expert_routes(x, idx, weights, w13, w2, first)
                def dense():
                    h13 = linear(x.unsqueeze(0).expand(local_e, t, h), w13)
                    act = silu_and_mul(h13.reshape(local_e * t, 2 * inter)).view(local_e, t, inter)
                    out = linear(act, w2)
                    wm = _local_route_weights(idx, weights, first, local_e)
                    return out * wm.to(dtype)[..., None]
                graph = None
                for change in range(4):
                    x.normal_()
                    ids = torch.rand(t, global_e, device='cuda').argsort(-1)[:, :k].contiguous()
                    if change == 1:  # all routes miss this rank
                        allowed = torch.tensor([e for e in range(global_e) if not first <= e < first + local_e], device='cuda')
                        ids = allowed[torch.rand(t, len(allowed), device='cuda').argsort(-1)[:, :k]]
                    if change == 2:  # every token hits the same local expert
                        ids[:, 0] = first
                        # avoid duplicate top-k by choosing remaining nonlocals
                        allowed = torch.tensor([e for e in range(global_e) if not first <= e < first + local_e], device='cuda')
                        ids[:, 1:] = allowed[:k-1]
                    idx.copy_(ids)
                    weights.copy_(torch.softmax(torch.randn(t, k, device='cuda'), -1).to(dtype))
                    expected, actual = dense(), sparse()
                    assert torch.equal(actual, expected), (dtype, h, t, change, (actual-expected).abs().max().item())
                    if graph is None:
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3): sparse()
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            replay = sparse()
                    graph.replay()
                    assert torch.equal(replay, expected), ('graph', dtype, h, t, change)
                    # Compaction order changes with permutation, and tile count
                    # changes with padding. Valid token results must be exact.
                    perm = torch.randperm(t, device='cuda')
                    permuted = sparse_expert_routes(x[perm], idx[perm], weights[perm], w13, w2, first)
                    assert torch.equal(permuted, actual[:, perm])
                    padded = sparse_expert_routes(torch.cat([x, x[:1]]), torch.cat([idx, idx[:1]]),
                        torch.cat([weights, weights[:1]]), w13, w2, first)
                    assert torch.equal(padded[:, :t], actual)
                    rows.append(dict(dtype=str(dtype), hidden=h, batch=t, change=change,
                                     dense_exact=True, graph_exact=True, permutation_exact=True, padding_exact=True))
                graph.reset()
    Path(args.output).write_text(json.dumps(dict(rows=rows, passed=len(rows)), indent=2))
    print('PASS', len(rows), flush=True)


if __name__ == '__main__':
    main()
