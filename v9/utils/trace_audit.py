"""Inspect real activity categories; a CUDA filename is not collection proof."""
from collections import Counter
import gzip
import json
from pathlib import Path


def audit_trace(path):
    path=Path(path)
    opener=gzip.open if path.suffix == '.gz' else open
    with opener(path,'rt',encoding='utf-8') as handle:
        data=json.load(handle)
    events=data.get('traceEvents',[])
    categories=Counter(e.get('cat','') for e in events if e.get('ph') == 'X')
    kernels=[e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    nccl=[e for e in kernels if any(name in e.get('name','').lower() for name in ('nccl','rccl'))]
    return dict(file=str(path), event_count=len(events), duration_categories=dict(categories),
        cpu_ops=categories['cpu_op'], cuda_runtime=categories['cuda_runtime'],
        gpu_kernels=len(kernels), nccl_kernels=len(nccl),
        cuda_streams=sorted({str(e.get('args',{}).get('stream',e.get('tid'))) for e in kernels}),
        module_labels=sorted({e.get('name','') for e in events
            if e.get('cat') == 'user_annotation' and '.layer' in e.get('name','')}),
        cuda_verified=bool(kernels),
        note='NCCL count uses kernel names; zero is not proof of CPU communication. '
             'No overlap or cross-rank clock correctness inferred by this audit.')
