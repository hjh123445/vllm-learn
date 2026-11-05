"""Capture complete worker forwards, independently of throughput measurement."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--pp', type=int, default=2)
    ap.add_argument('--enable-expert-parallel', action='store_true')
    ap.add_argument('--mode', choices=['baseline', 'optimized'], default='optimized')
    ap.add_argument('--workload', default='short64')
    ap.add_argument('--requests', type=int, default=8)
    ap.add_argument('--messages', type=int, default=4)
    ap.add_argument('--start-message', type=int, default=1)
    ap.add_argument('--microbatch-tokens', type=int, default=128)
    ap.add_argument('--max-kv-blocks', type=int, default=None)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=32)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for key, value in dict(VLLM_LEARN_TRACE_DIR=str(out/'trace'),
            VLLM_LEARN_CUDA_PROFILE='1', VLLM_LEARN_PROFILE_MODE='window',
            VLLM_LEARN_PROFILE_START_MESSAGE=str(args.start_message),
            VLLM_LEARN_PROFILE_MESSAGES=str(args.messages),
            VLLM_LEARN_METRICS_DIR=str(out)).items():
        os.environ[key] = value
    from engine.core import EngineCore
    from utils.observed_runner import ObservedModelRunner, ObservedPipelineTransport
    from samplingParams import SamplingParams
    rows = json.loads(Path(args.inputs).read_text())[args.workload][:args.requests]
    (out/'config.json').write_text(json.dumps(dict(vars(args), profiled=True,
        note='GPU synchronization and trace export change scheduling; exclude from throughput tables'), indent=2))
    (out/'inputs.json').write_text(json.dumps(rows))
    core = EngineCore(args.model, tp_size=args.tp, enable_expert_parallel=args.enable_expert_parallel, pp_size=args.pp, master_port=31136,
        runner_cls=ObservedModelRunner, transport_cls=ObservedPipelineTransport,
        pipeline_microbatch_tokens=args.microbatch_tokens, pipeline_lanes=2,
        pipeline_mode=args.mode, config_kwargs=dict(max_num_seqs=128, max_model_len=8192,
        max_num_batched_tokens=512, gpu_memory_utilization=.70, enforce_eager=True,
        max_kvcache_blocks=args.max_kv_blocks,
        enable_prefix_caching=False))
    try:
        handles = [core.add_request(row['input_ids'], SamplingParams(temperature=0.,
                   max_tokens=args.max_tokens, ignore_eos=True)) for row in rows]
        while not core.is_finished():
            core.step()
        (out/'outputs.json').write_text(json.dumps([s.completion_token_ids for s in handles]))
    finally:
        core.shutdown()


if __name__ == '__main__':
    main()
