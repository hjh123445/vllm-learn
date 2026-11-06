"""Validate the opt-in automatic native grouped-GEMM policy on CUDA.

The loop remains the reference path. ``auto`` must select exactly the same
implementation as explicit ``torch_grouped`` on CUDA BF16, and must fall back
to ``loop`` for unsupported CPU/dtype inputs. This component oracle is not a
whole-model accuracy benchmark.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    import torch
    from layers.moe import GroupedExperts

    if not torch.cuda.is_available():
        raise RuntimeError('This validation needs a CUDA device')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(9137)

    module = GroupedExperts(num_experts=16, hidden_size=64, intermediate=32)
    module = module.to(device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        module.w13.normal_(std=0.02)
        module.w2.normal_(std=0.02)

    cpu_probe = torch.empty((2, 64), dtype=torch.float32)
    module.backend = 'auto'
    assert module.resolve_backend(cpu_probe) == 'loop'

    records = []
    with torch.inference_mode():
        for tokens in (1, 8, 64, 512):
            x = torch.randn((tokens, 64), device='cuda', dtype=torch.bfloat16)
            # Deliberately skewed route counts leave experts empty and exercise
            # uneven grouped segments, as happens in real MoE batches.
            route = torch.arange(tokens * 4, device='cuda').reshape(tokens, 4)
            ids = ((route * 7) % 11).long()
            weights = torch.softmax(torch.randn(tokens, 4, device='cuda'), -1).to(x.dtype)

            module.backend = 'loop'
            reference = module(x, ids, weights)
            module.backend = 'torch_grouped'
            explicit = module(x, ids, weights)
            module.backend = 'auto'
            selected = module.resolve_backend(x)
            automatic = module(x, ids, weights)

            assert selected == 'torch_grouped', selected
            # auto and explicit must route through the same grouped API and be
            # bit-identical; the loop is checked with a BF16 numerical bound.
            assert torch.equal(automatic, explicit), (tokens, 'auto != explicit grouped')
            delta = automatic.float() - reference.float()
            max_abs = float(delta.abs().max().item())
            rmse = float(delta.square().mean().sqrt().item())
            passed = torch.allclose(automatic, reference, atol=0.015, rtol=0.04)
            assert passed, (tokens, max_abs, rmse)
            records.append(dict(tokens=tokens, routes=tokens * 4,
                selected_backend=selected, auto_equals_explicit=torch.equal(automatic, explicit),
                loop_max_abs=max_abs, loop_rmse=rmse, passed=passed))

    result = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        configured_backend='auto', cpu_fallback='loop', records=records,
        scope='CUDA BF16 grouped expert component oracle; not full-model HF parity')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
