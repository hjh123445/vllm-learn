"""Scoped workaround for CUDA13/PyTorch Kineto empty activity buffers.

Only used by the independent capture tool, after device synchronization and
before profiler.stop. Does not install/replace global CUDA libraries.
"""
import ctypes
import os
from pathlib import Path
import torch


def flush_cuda_activity():
    mode=os.getenv('VLLM_LEARN_CUPTI_FLUSH','auto')
    if mode not in ('auto','on','off'):
        raise ValueError('VLLM_LEARN_CUPTI_FLUSH must be auto, on or off')
    version=torch.version.cuda or ''
    if mode=='off' or not torch.cuda.is_available() or (mode=='auto' and not version.startswith('13.0')):
        return dict(attempted=False)
    major=version.split('.')[0]
    root=Path(torch.__file__).resolve().parent.parent
    candidates=[root/'nvidia'/f'cu{major}'/'lib'/f'libcupti.so.{major}',
                root/'nvidia'/'cuda_cupti'/'lib'/f'libcupti.so.{major}']
    # Load only an existing installed library of the matching CUDA major.
    library=next((p for p in candidates if p.is_file()),None)
    if library is None:
        return dict(attempted=False,error='matching installed CUPTI library not located')
    handle=ctypes.CDLL(str(library))
    flush=handle.cuptiActivityFlushAll
    flush.argtypes=[ctypes.c_uint32]
    flush.restype=ctypes.c_int
    code=flush(1)  # CUPTI_ACTIVITY_FLAG_FLUSH_FORCED; GPU is already drained.
    if code:
        raise RuntimeError(f'CUPTI activity flush failed: status={code}, library={library}')
    return dict(attempted=True,status=code,library=str(library))
