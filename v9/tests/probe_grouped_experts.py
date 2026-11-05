"""CUDA BF16 uneven/empty expert oracle and native grouped-GEMM latency.

Uses synthetic weights only. Absolute times include routing and launch costs,
not communication; they cannot substitute for whole-model throughput.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    import torch
    from layers.moe import GroupedExperts
    torch.set_num_threads(4)
    torch.cuda.set_device(2)
    torch.manual_seed(931)
    model = GroupedExperts(16, 2048, 768).to(device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        model.w13.normal_(std=.02)
        model.w2.normal_(std=.02)
    records = []
    with torch.inference_mode():
        for tokens in (0, 8, 64, 512):
            x = torch.randn(tokens, 2048, device='cuda', dtype=torch.bfloat16)
            # Eight routes/token; skew excludes several experts completely.
            ids = ((torch.arange(tokens*8, device='cuda').reshape(tokens,8)*7) % 11).long()
            w = torch.softmax(torch.randn(tokens,8,device='cuda'),dim=-1).to(x.dtype)
            model.backend = 'loop'
            reference = model(x,ids,w)
            model.backend = 'torch_grouped'
            candidate = model(x,ids,w)
            difference = candidate.float()-reference.float()
            max_abs = difference.abs().max().item() if tokens else 0.
            rmse = difference.square().mean().sqrt().item() if tokens else 0.
            passed = torch.allclose(candidate,reference,atol=.015,rtol=.04)
            assert passed, (tokens,max_abs,rmse)
            times = {}
            for backend in ('loop','torch_grouped'):
                model.backend = backend
                for _ in range(5):model(x,ids,w)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(30):model(x,ids,w)
                torch.cuda.synchronize()
                times[backend] = (time.perf_counter()-start)*1000/30
            records.append(dict(tokens=tokens,routes=tokens*8,max_abs=max_abs,rmse=rmse,
                atol=.015,rtol=.04,passed=passed,latency_ms=times))
        profiles = {}
        if args.profile:
            for backend in ('loop', 'torch_grouped'):
                model.backend = backend
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with torch.autograd.profiler.record_function('expert_backend.'+backend):
                        model(x,ids,w)
                    torch.cuda.synchronize()
                path = Path(args.output).with_name('component_'+backend+'_trace.json')
                path.parent.mkdir(parents=True,exist_ok=True)
                prof.export_chrome_trace(str(path))
                events = json.loads(path.read_text())['traceEvents']
                from collections import Counter
                profiles[backend] = dict(cpu_ops=dict(Counter(e['name'] for e in events
                    if e.get('cat')=='cpu_op')), kernels=dict(Counter(e['name'] for e in events
                    if e.get('cat')=='kernel')))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),records=records,profiles=profiles,
        scope='BF16 synthetic loop-vs-native comparison; not full model/HF parity')
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
