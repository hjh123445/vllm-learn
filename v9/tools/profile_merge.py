"""Overlay per-rank Chrome traces; never mix synthetic and Kineto clocks.

python v9/tools/profile_merge.py --input-dir trace --same-host --output full.json.gz
Only combine workers from ONE host and ONE run. No PyTorch dependency.
"""
import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.trace_io import TraceReader, trace_paths

GPU_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersection_length(a, b):
    a, b = union(a), union(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize(events):
    """GPU interval unions, not sums of nested CPU scopes or kernels."""
    groups = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') not in GPU_CATEGORIES:
            continue
        group = groups[(event['args']['trace_source'], event['pid'])]
        interval = (event['ts'], event['ts'] + event.get('dur', 0))
        group['all'].append(interval)
        if event['cat'] == 'kernel':
            kind = 'nccl' if 'nccl' in event['name'].lower() else 'non_nccl'
            group[kind].append(interval)
    rows = []
    for (source, pid), group in groups.items():
        lengths = {k: sum(b-a for a, b in union(v)) for k, v in group.items()}
        span = max(b for _, b in group['all']) - min(a for a, _ in group['all'])
        overlap = intersection_length(group['nccl'], group['non_nccl'])
        rows.append({
            'source': source, 'pid': pid, 'observed_gpu_span_us': span,
            'gpu_active_union_us': lengths['all'],
            'non_nccl_kernel_union_us': lengths.get('non_nccl', 0),
            'nccl_kernel_union_us': lengths.get('nccl', 0),
            'nccl_non_nccl_intersection_us': overlap,
            'fraction_nccl_time_overlapped': overlap/lengths['nccl'] if lengths.get('nccl') else None,
            'observed_inactive_fraction': 1-lengths['all']/span if span else None,
        })
    return {'event_count': len(events), 'gpu_tracks': rows,
            'caveat': 'GPU inactivity within each captured GPU span is NOT a pipeline bubble metric. '
                      'NCCL kernels include waiting; non-NCCL kernels include routing. '
                      'No steady-state utilization or cross-host clock accuracy is inferred.'}


def merge_traces(paths, *, same_host=False, shared_timestamps=False, sink=None):
    if not paths:
        raise ValueError('No input traces')
    if len(paths) > 1 and not same_host:
        raise ValueError('Alignment requires --same-host (one run); cross-host clocks need external calibration')
    sources = []
    for path in map(Path, paths):
        reader = TraceReader(path)
        first_ts = None
        active, gpu = set(), set()
        scope = ''
        for event in reader.events():
            if event.get('ph') != 'M':
                active.add(event.get('pid', 0))
                if 'ts' in event:
                    first_ts = min(first_ts, event['ts']) if first_ts is not None else event['ts']
            if event.get('cat') in GPU_CATEGORIES:
                gpu.add(event.get('pid', 0))
            if not scope and re.search(r'(?:^|\.)stage\d+\.', event.get('name', '')):
                scope = event['name']
        sources.append((path, reader.metadata, first_ts, active, gpu, scope))
    clocks = []
    for path, data, *_ in sources:
        if data.get('metadata', {}).get('clock') == 'host_monotonic_us':
            clocks.append('host_monotonic_us')
        elif data.get('baseTimeNanoseconds') is not None:
            clocks.append('kineto')
        else:
            clocks.append('unspecified')
    if len(set(clocks)) != 1:
        raise ValueError('Mixed clock domains: merge cuda_rank and pipeline_rank separately')
    if clocks[0] == 'unspecified' and len(paths) > 1 and not shared_timestamps:
        raise ValueError('Missing time bases: --shared-timestamps requires a verified shared microsecond clock')
    bases = [int(d.get('baseTimeNanoseconds', 0)) for _, d, *_ in sources]
    base = min(bases)
    offsets = [(b-base)/1000 for b in bases]
    timestamps = [first+offset for (_, _, first, *_), offset in zip(sources, offsets) if first is not None]
    origin = min(timestamps, default=0)
    result, manifest, gpu_result = [], [], []
    event_count = 0

    def emit(event):
        nonlocal event_count
        event_count += 1
        if sink is None:
            result.append(event)
        else:
            sink(event)
            if event.get('cat') in GPU_CATEGORIES:
                gpu_result.append(event)
    next_pid = next_tid = next_link = 1
    for source_index, ((path, data, _, active_pids, gpu_pids, scope), offset) in enumerate(zip(sources, offsets)):
        events = TraceReader(path).events()
        match = re.search(r'rank(\d+)', path.name)
        # Independent DP worlds reuse distributed rank zero; the exporter
        # namespaces filenames over frontend replicas, which must take priority.
        rank = data.get('metadata', {}).get('rank')
        if rank is None:
            rank = int(match[1]) if match else data.get('distributedInfo', {}).get('rank')
        stage = data.get('metadata', {}).get('pp_stage')
        if stage is None:
            found = re.search(r'(?:^|\.)stage(\d+)', scope)
            stage = int(found[1]) if found else None
        prefix = f"rank{rank if rank is not None else '?'}"
        if stage is not None:
            prefix += f' / stage{stage}'
        prefix += f' / source{source_index}'
        pids, tids, links = {}, {}, {}

        def pid_for(old):
            nonlocal next_pid
            if old not in pids:
                pids[old], next_pid = next_pid, next_pid+1
            return pids[old]

        def tid_for(pid, tid):
            nonlocal next_tid
            if (pid, tid) not in tids:
                tids[pid, tid], next_tid = next_tid, next_tid+1
            return tids[pid, tid]

        def link_for(family, value):
            nonlocal next_link
            key = (family, json.dumps(value, sort_keys=True))
            if key not in links:
                links[key], next_link = next_link, next_link+1
            return links[key]

        gpu_count = 0
        for original in events:
            old_pid, old_tid = original.get('pid', 0), original.get('tid', 0)
            if original.get('ph') == 'M' and old_pid not in active_pids:
                continue  # Kineto also emits metadata for unused visible GPUs.
            event = original
            event['pid'], event['tid'] = pid_for(old_pid), tid_for(old_pid, old_tid)
            if 'ts' in event:
                event['ts'] = event['ts'] + offset - origin
            args = event.setdefault('args', {})
            args['trace_source'] = path.name
            for field in ('External id', 'correlation', 'Ev Idx', 'Record function id'):
                if field in args:
                    args[field] = link_for(field, args[field])
            for field in ('id', 'bind_id'):
                if field in event:
                    event[field] = link_for('flow', event[field])
            if 'id2' in event:
                event['id2'] = {k: link_for('flow', v) for k, v in event['id2'].items()}
            if event.get('cat') in GPU_CATEGORIES:
                gpu_count += 1
            if event.get('ph') == 'M' and event.get('name') == 'process_name':
                args['name'] = f"{prefix} / {'GPU device' if old_pid in gpu_pids else 'CPU pid'} {old_pid}"
            if event.get('ph') == 'M' and event.get('name') == 'thread_name':
                args['name'] = f"{args.get('name', old_tid)} [original tid={old_tid}]"
            emit(event)
        for old_pid in sorted(active_pids, key=str):
            emit({'ph': 'M', 'name': 'process_name', 'pid': pid_for(old_pid), 'tid': 0,
                           'args': {'name': f"{prefix} / {'GPU device' if old_pid in gpu_pids else 'CPU pid'} {old_pid}"}})
        for (old_pid, old_tid), new_tid in list(tids.items()):
            if old_pid in gpu_pids:
                emit({'ph': 'M', 'name': 'thread_name', 'pid': pid_for(old_pid), 'tid': new_tid,
                               'args': {'name': f'CUDA stream {old_tid}'}})
        manifest.append({'file': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'rank': rank, 'stage': stage, 'base_ns': bases[source_index],
                         'original_process_ids': {str(k): v for k, v in pids.items()},
                         'gpu_event_count': gpu_count,
                         'source_metadata': {k: v for k, v in data.items() if k != 'traceEvents'}})
    return {'displayTimeUnit': 'ms', 'traceEvents': result,
            'metadata': {'clock_domain': clocks[0], 'same_host_asserted': same_host,
                         'common_base_ns': base, 'common_origin_us': origin, 'timestamp_unit': 'us',
                         'alignment': 'ts_out=ts_in+(base_in-common_base)/1000-common_origin; dur unchanged',
                         'workers': manifest, 'summary': dict(summarize(result if sink is None else gpu_result), event_count=event_count)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-dir', type=Path)
    inputs.add_argument('--inputs', nargs='+', type=Path)
    parser.add_argument('--kind', choices=('cuda', 'pipeline'), default='cuda')
    parser.add_argument('--same-host', action='store_true')
    parser.add_argument('--shared-timestamps', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    paths = args.inputs or (trace_paths(args.input_dir) if args.kind == 'cuda' else sorted(args.input_dir.glob('pipeline_rank*.json')))
    if args.output.resolve() in [p.resolve() for p in paths]:
        parser.error('output must differ from every input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.partial')
    opener = gzip.open if args.output.suffix == '.gz' else open
    with opener(temporary, 'wt', encoding='utf-8') as handle:
        handle.write('{"displayTimeUnit":"ms","traceEvents":[')
        count = 0

        def write_event(event):
            nonlocal count
            if count:
                handle.write(',')
            json.dump(event, handle, ensure_ascii=False, separators=(',', ':'))
            count += 1

        result = merge_traces(paths, same_host=args.same_host,
                              shared_timestamps=args.shared_timestamps, sink=write_event)
        handle.write('],"metadata":')
        json.dump(result['metadata'], handle, ensure_ascii=False, separators=(',', ':'))
        handle.write('}')
    temporary.replace(args.output)
    print(json.dumps(result['metadata']['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
