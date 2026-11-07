"""Shared, streaming trace acceptance; presence alone is never completeness."""
import json
from pathlib import Path


def coverage_errors(row, capture=None, *, require_cuda=True):
    """Compare independently recorded forwards/layers with CPU and GPU coverage."""
    errors = []
    messages = {m['message']: m for m in row['message_gpu']}
    if capture is not None and not all(key in capture for key in
            ('stage', 'rank', 'layer_range', 'captured_messages', 'executed_messages', 'start_message')):
        errors.append('legacy/incomplete capture manifest lacks execution inventory or layer range')
        capture = None
    if capture is not None and capture['rank'] != row['rank']:
        errors.append('capture manifest rank does not match trace filename')
    if capture is None:
        errors.append('missing capture manifest; executed-message inventory cannot be verified')
        expected = list(messages)
        components = {name for m in messages.values() for name in m['cpu_components']}
    else:
        if capture.get('closed_cleanly') is not True:
            errors.append('capture did not close cleanly')
        stage = capture['stage']
        start, end = capture['layer_range']
        components = {f'stage{stage}.layer{layer}.{op}' for layer in range(start, end)
                      for op in ('attention', 'moe')}
        expected = []
        for m in capture['captured_messages']:
            name = (f"rank{capture['rank']}.stage{stage}.message{m['message_index']}."
                    f"mb{m['microbatch_id']}.{'prefill' if m['is_prefill'] else 'decode'}")
            expected.append(name)
            if m.get('succeeded') is not True:
                errors.append(f'{name}: forward failed or completion unknown')
        first = capture['start_message']
        stop = capture['executed_messages'] + 1
        if capture.get('requested_messages', 0):
            stop = min(stop, first + capture['requested_messages'])
        indices = [m['message_index'] for m in capture['captured_messages']]
        if indices != list(range(first, stop)):
            errors.append('message inventory has gaps')
        extra = set(messages) - set(expected)
        if extra:
            errors.append(f'unexpected CPU message ranges: {sorted(extra)}')
    if not expected:
        errors.append('no captured forwards; no model execution coverage to verify')
    for name in expected:
        message = messages.get(name)
        if message is None:
            errors.append(f'{name}: missing entire CPU/GPU message range')
            continue
        required = components
        if capture is not None:
            entry = capture['captured_messages'][expected.index(name)]
            # Idle DP workers intentionally service MoE only; no local attention.
            if entry.get('scheduled_sequences') == 0:
                required = {c for c in components if c.endswith('.moe')}
            if entry.get('graph_replays', 0):
                # Replay bypasses Python modules: certify launch and GPU
                # activity, but never invent per-layer Python/GPU ranges.
                required = set()
                if message.get('graph_launches', 0) != entry['graph_replays']:
                    errors.append(f'{name}: graph launch count disagrees with executed replays')
        for side in ('cpu', 'gpu') if require_cuda else ('cpu',):
            missing = required - set(message[f'{side}_components'])
            if missing:
                errors.append(f'{name}: missing {side.upper()} components {sorted(missing)}')
        if require_cuda and not message['kernel_count']:
            errors.append(f'{name}: zero attributed GPU kernels')
    if require_cuda and row['unmatched_kernel_launches']:
        errors.append(f"kernel launches without GPU activity: {row['unmatched_kernel_launches']}")
    return errors


def audit_trace(path, capture=None, *, require_cuda=None):
    from tools.profile_analyze import analyze
    path = Path(path)
    row, _ = analyze(path)
    if capture is None:
        manifest = path.parent / f"capture_rank{row['rank']}.json"
        if manifest.exists():
            capture = json.loads(manifest.read_text(encoding='utf-8'))
    if require_cuda is None:
        require_cuda = capture.get('cuda_activities', True) if capture else True
    errors = coverage_errors(row, capture, require_cuda=require_cuda)
    counts = row['categories']
    all_graph = bool(capture and capture['captured_messages']) and all(
        m.get('graph_replays', 0) for m in capture['captured_messages'])
    if not counts.get('cpu_op') or (not row['module_labels'] and not all_graph):
        errors.append('missing ATen or module ranges')
    if require_cuda and (not row['gpu_kernels'] or not
                        (counts.get('cuda_runtime') or counts.get('cuda_driver'))):
        errors.append('missing actual CUDA kernels or launch APIs')
    return dict(file=str(path), duration_categories=counts,
        event_count=sum(counts.values()), cpu_ops=counts.get('cpu_op', 0),
        cuda_runtime=counts.get('cuda_runtime', 0), gpu_kernels=row['gpu_kernels'],
        nccl_kernels=row['nccl_kernels'], cuda_streams=row['cuda_streams'],
        module_labels=row['module_labels'], cuda_verified=bool(row['gpu_kernels']),
        completeness_verified=not errors, errors=errors, message_coverage=row['message_gpu'],
        layer_attribution_verified=not errors and not any(
            m.get('graph_replays', 0) for m in (capture or {}).get('captured_messages', [])),
        note='Checks executed-message inventory, CPU/GPU Attention+MoE coverage and launch correlation; '
             'not proof of zero dropped events, numerical correctness or performance.')


def validate_traces(directory, *, expected_ranks, require_cuda=True, require_nccl=False):
    """Persist an aggregate result; reject duplicate/missing/stale rank artifacts."""
    from tools.trace_io import trace_paths
    import re
    directory = Path(directory)
    audits, errors = [], []
    try:
        paths = trace_paths(directory)
        ranks = {int(re.search(r'rank(\d+)', p.name)[1]) for p in paths}
        if ranks != set(range(expected_ranks)):
            errors.append(f'rank set {sorted(ranks)} != expected {list(range(expected_ranks))}')
        for path in paths:
            audit = audit_trace(path, require_cuda=require_cuda)
            audits.append(audit)
            errors.extend(f'{path.name}: {e}' for e in audit['errors'])
            if require_nccl and not audit['nccl_kernels']:
                errors.append(f'{path.name}: no NCCL GPU kernel')
    except (ValueError, OSError, KeyError) as error:
        errors.append(str(error))
    summary = dict(passed=not errors, ranks=len(audits), errors=errors, audits=audits)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'trace_validation.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def finish_profile(output, *, expected_ranks):
    summary = validate_traces(Path(output) / 'trace', expected_ranks=expected_ranks)
    if not summary['passed']:
        raise RuntimeError(f"Profile collection incomplete; see {Path(output) / 'trace/trace_validation.json'}")
    print(f'Profile coverage verified for {expected_ranks} ranks: {Path(output) / "trace"}', flush=True)
