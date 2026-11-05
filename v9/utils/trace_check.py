"""Validate CPU/ATen, module labels and actual CUDA activities for each rank."""
import argparse
import json
from pathlib import Path
from utils.trace_audit import audit_trace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True,type=Path)
    parser.add_argument('--require-cuda',action='store_true')
    parser.add_argument('--require-nccl',action='store_true')
    parser.add_argument('--expected-ranks',type=int)
    args=parser.parse_args()
    paths=sorted(args.input_dir.glob('cuda_rank*.json'))+sorted(args.input_dir.glob('cuda_rank*.json.gz'))
    if not paths:
        parser.error('no cuda_rank traces found')
    audits=[audit_trace(p) for p in paths]
    passed=all(a['cpu_ops'] > 0 and bool(a['module_labels'])
        and (not args.require_cuda or a['gpu_kernels'] > 0)
        and (not args.require_nccl or a['nccl_kernels'] > 0) for a in audits)
    if args.expected_ranks is not None:
        passed=passed and len(paths)==args.expected_ranks
    summary=dict(passed=passed,ranks=len(audits),audits=audits)
    (args.input_dir/'trace_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not passed:
        raise SystemExit(1)
