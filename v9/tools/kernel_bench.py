"""CUDA-event microbenchmark for the fused MoE SiLU×up kernel."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(fn, warmup, iterations, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "p95_us": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "samples_us": samples,
    }


def measure_graph(fn, warmup, iterations, repeats):
    graph_iterations = min(iterations, 16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iterations):
            fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / graph_iterations)
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "p95_us": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "samples_us": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true", help="export a separate single-op CPU/CUDA trace")
    args = parser.parse_args()
    if min(args.tokens + [args.intermediate, args.warmup, args.iterations, args.repeats]) < 1:
        parser.error('sizes and repetitions must be positive')
    if args.profile and args.output is None:
        parser.error('--profile requires --output')
    global torch
    import torch
    import torch.nn.functional as F
    from layers.kernels import silu_and_mul
    if not torch.cuda.is_available():
        raise RuntimeError("bench_silu_kernel requires CUDA")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    records = []
    torch.manual_seed(731)
    for tokens in args.tokens:
        x = torch.randn((tokens, 2 * args.intermediate), device="cuda", dtype=dtype)
        gate, up = x.chunk(2, dim=-1)
        reference = F.silu(gate) * up
        fused = silu_and_mul(x, backend="triton")
        delta = fused.float() - reference.float()
        atol = {torch.float32: 2e-6, torch.float16: 2e-3,
                torch.bfloat16: 2e-2}[dtype]
        torch.testing.assert_close(fused, reference, atol=atol, rtol=atol)

        torch_fn = lambda: silu_and_mul(x, backend="torch")
        triton_fn = lambda: silu_and_mul(x, backend="triton")
        torch_timing = measure(torch_fn,
                               args.warmup, args.iterations, args.repeats)
        triton_timing = measure(triton_fn,
                                args.warmup, args.iterations, args.repeats)
        torch_graph = measure_graph(torch_fn,
                                    args.warmup, args.iterations, args.repeats)
        triton_graph = measure_graph(triton_fn,
            args.warmup, args.iterations, args.repeats)
        records.append({
            "tokens": tokens,
            "intermediate": args.intermediate,
            "dtype": args.dtype,
            "max_abs_error": float(delta.abs().max().item()),
            "rmse": float(delta.square().mean().sqrt().item()),
            "torch": torch_timing,
            "triton": triton_timing,
            "speedup": torch_timing["median_us"] / triton_timing["median_us"],
            "cuda_graph_torch_us": torch_graph["median_us"],
            "cuda_graph_torch_p95_us": torch_graph["p95_us"],
            "cuda_graph_torch_samples_us": torch_graph["samples_us"],
            "cuda_graph_triton_us": triton_graph["median_us"],
            "cuda_graph_triton_p95_us": triton_graph["p95_us"],
            "cuda_graph_triton_samples_us": triton_graph["samples_us"],
            "cuda_graph_speedup": torch_graph["median_us"] / triton_graph["median_us"],
        })

        if args.profile:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(16):
                    torch_fn()
                    triton_fn()
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                for name, fn in [('torch.eager', torch_fn), ('triton.eager', triton_fn),
                                 ('both.graph', graph.replay)]:
                    with torch.profiler.record_function(f'silu.tokens{tokens}.{name}'):
                        for _ in range(8):
                            fn()
                torch.cuda.synchronize()
                from utils.cupti_flush import flush_cuda_activity
                flush_cuda_activity()
            trace = args.output.with_name(f'{args.output.stem}.tokens{tokens}.trace.json')
            trace.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace))

    result = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "comparison": "matched silu_and_mul wrapper for BOTH backends, including chunk/view/validation/allocation; fixed seed731",
        "note": "eager CUDA-event timing includes host launch gaps; CUDA Graph timing is after replay warmup; both are isolated operator timings and exclude GEMM, routing, communication, and full-model scheduling",
        "records": records,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
