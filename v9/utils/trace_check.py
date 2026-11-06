"""Validate CPU/ATen, module labels and actual CUDA activities for each rank."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.trace_audit import validate_traces
from tools.trace_io import trace_paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True,type=Path)
    parser.add_argument('--require-cuda',action='store_true')
    parser.add_argument('--require-nccl',action='store_true')
    parser.add_argument('--expected-ranks',type=int)
    args=parser.parse_args()
    expected=args.expected_ranks
    if expected is None:
        try:
            expected=len(trace_paths(args.input_dir))
        except ValueError as error:
            parser.error(str(error))
    summary=validate_traces(args.input_dir,expected_ranks=expected,
        require_cuda=args.require_cuda or args.require_nccl,require_nccl=args.require_nccl)
    print(json.dumps({k:v for k,v in summary.items() if k!='audits'},ensure_ascii=False,indent=2))
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
