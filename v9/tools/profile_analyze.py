"""Audit ranks and CPU/CUDA/NCCL coverage; quantify layers, gaps and overlap."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.trace_io import TraceReader, duration, intersection, trace_paths, union

LAYER = re.compile(r'stage(\d+)\.layer\.?(\d+)\.(attention|moe)$')
MESSAGE = re.compile(r'(?:rank\d+\.stage\d+\.message\d+\.mb\d+|stage\d+\.microbatch\d+\.lane\d+)\.(prefill|decode)$')


def analyze(path):
    scopes = defaultdict(list)
    message_scopes = defaultdict(list)
    grouped_scopes = defaultdict(list)
    counts, cpu_names = Counter(), Counter()
    messages, layers, module_labels = [], set(), set()
    reader = TraceReader(path)
    for event in reader.events():
        if event.get('ph') != 'X':
            continue
        cat, name = event.get('cat', ''), event.get('name', '')
        counts[cat] += 1
        if cat == 'user_annotation' and '.layer' in name:
            module_labels.add(name)
        if cat == 'cpu_op':
            cpu_names[name] += 1
            if name == 'aten::_grouped_mm':
                grouped_scopes[(event.get('pid'),event.get('tid'))].append((event['ts'],event['ts']+event['dur']))
        match = LAYER.search(name) if cat == 'user_annotation' else None
        if match:
            key = tuple(match.groups())
            layers.add(key)
            scopes[(event.get('pid'), event.get('tid'))].append((event['ts'], event['ts']+event['dur'], key))
        if cat == 'user_annotation' and MESSAGE.search(name):
            messages.append(dict(name=name, ts=event['ts'], dur=event.get('dur', 0)))
            message_scopes[(event.get('pid'),event.get('tid'))].append((event['ts'],event['ts']+event.get('dur',0),name))
    for ranges in scopes.values():
        ranges.sort()
    starts = {key: [r[0] for r in ranges] for key, ranges in scopes.items()}
    # Older traces nest a legacy lane range inside the modern worker range.
    # Attribute to the worker message once, rather than report its wrapper as empty.
    aliases = set()
    for key, ranges in message_scopes.items():
        modern = [(a, b) for a, b, name in ranges if name.startswith('rank')]
        aliases.update(name for a, b, name in ranges if not name.startswith('rank')
                       and any(start <= a and b <= end for start, end in modern))
        message_scopes[key] = sorted(r for r in ranges if r[2] not in aliases)
    messages = [m for m in messages if m['name'] not in aliases]
    message_starts={key:[r[0] for r in ranges] for key,ranges in message_scopes.items()}
    for ranges in grouped_scopes.values():
        ranges.sort()
    grouped_starts={key:[a for a,b in ranges] for key,ranges in grouped_scopes.items()}
    grouped_children=Counter()
    ext, corr, kernels, copies = {}, {}, [], []
    ext_message,corr_message={},{}
    api = Counter()
    launches = set()
    for event in TraceReader(path).events():
        if event.get('ph') != 'X':
            continue
        cat, args = event.get('cat'), event.get('args', {})
        if cat in ('cpu_op', 'cuda_runtime', 'cuda_driver'):
            if cat in ('cuda_runtime', 'cuda_driver'):
                api[event.get('name', '')] += event.get('dur', 0)
            key = (event.get('pid'), event.get('tid'))
            if cat == 'cpu_op' and event.get('name') in ('aten::mm','aten::item','aten::_local_scalar_dense'):
                index=bisect_right(grouped_starts.get(key,[]),event['ts'])-1
                if index>=0 and event['ts']<grouped_scopes[key][index][1]:
                    grouped_children[event['name']]+=1
            index = bisect_right(starts.get(key, []), event['ts'])-1
            if index >= 0:
                a, b, label = scopes[key][index]
                if a <= event['ts'] < b:
                    if 'External id' in args:
                        ext[args['External id']] = label
                    if 'correlation' in args:
                        corr[args['correlation']] = label
            index=bisect_right(message_starts.get(key,[]),event['ts'])-1
            if index>=0:
                a,b,message=message_scopes[key][index]
                if a<=event['ts']<b:
                    if 'External id' in args:
                        ext_message[args['External id']]=message
                    if 'correlation' in args:
                        corr_message[args['correlation']]=message
                    if (cat in ('cuda_runtime', 'cuda_driver') and 'Launch' in event.get('name', '')
                            and 'Kernel' in event.get('name', '') and 'correlation' in args):
                        launches.add(args['correlation'])
        if cat == 'kernel':
            kernels.append((event.get('name', ''), event['ts'], event['ts']+event.get('dur', 0),
                args.get('stream', event.get('tid')), args.get('External id'), args.get('correlation')))
        if cat in ('gpu_memcpy', 'gpu_memset'):
            copies.append((event['ts'], event['ts']+event.get('dur', 0)))
    compute, nccl = [], []
    by_layer = defaultdict(list)
    layer_compute, layer_nccl = defaultdict(list), defaultdict(list)
    by_message=defaultdict(lambda:dict(kernels=0,layers=set(),attention=[],moe=[],nccl=[],other=[]))
    cpu_components, gpu_components = defaultdict(set), defaultdict(set)
    # Seed from CPU messages: a message with ZERO GPU events must remain visible.
    for thread, ranges in message_scopes.items():
        for a, b, name in ranges:
            by_message[name]
            local = scopes.get(thread, [])
            first = bisect_left(starts.get(thread, []), a)
            last = bisect_left(starts.get(thread, []), b)
            cpu_components[name].update(label for start, end, label in local[first:last])
    hot_count, hot_time = Counter(), Counter()
    linked = 0
    for name, a, b, stream, external, correlation in kernels:
        (nccl if 'nccl' in name.lower() or 'rccl' in name.lower() else compute).append((a,b))
        hot_count[name] += 1
        hot_time[name] += b-a
        label = corr.get(correlation) or ext.get(external)
        message=corr_message.get(correlation) or ext_message.get(external)
        if message:
            bucket=by_message[message]
            bucket['kernels']+=1
            if label:
                bucket['layers'].add(int(label[1]))
                bucket[label[2]].append((a,b))
                gpu_components[message].add(label)
            else:
                bucket['other'].append((a,b))
            if 'nccl' in name.lower():
                bucket['nccl'].append((a,b))
        if label:
            linked += 1
            by_layer[label].append((a,b))
            (layer_nccl if 'nccl' in name.lower() or 'rccl' in name.lower() else layer_compute)[label].append((a,b))
    intervals = union(compute+nccl+copies)
    span = intervals[-1][1]-intervals[0][0] if intervals else 0
    gaps = [(intervals[i+1][0]-b, b) for i, (_, b) in enumerate(intervals[:-1])]
    layer_rows = [dict(stage=int(s), layer=int(l), operation=op,
        gpu_kernel_union_us=duration(v), kernel_count=len(v),
        non_nccl_union_us=duration(layer_compute[(s,l,op)]),
        nccl_union_us=duration(layer_nccl[(s,l,op)])) for (s,l,op),v in sorted(by_layer.items())]
    overlap = intersection(compute,nccl)
    stages = sorted({int(s) for s,_,_ in layers})
    row = dict(file=str(path), rank=int(re.search(r'rank(\d+)', path.name)[1]), stages=stages,
        categories=dict(counts), module_labels=sorted(module_labels),
        cuda_streams=sorted({str(k[3]) for k in kernels}),
        layer_labels=len(layers), global_layers=sorted({int(l) for _,l,_ in layers}), messages=messages,
        gpu_kernels=len(kernels), nccl_kernels=len(nccl), attributed_kernels=linked,
        attribution_fraction=linked/len(kernels) if kernels else 0,
        gpu_observed_span_us=span, gpu_activity_union_us=duration(intervals),
        observed_inactive_fraction=1-duration(intervals)/span if span else None,
        non_nccl_union_us=duration(compute), nccl_union_us=duration(nccl),
        compute_nccl_overlap_us=overlap,
        nccl_overlap_fraction=overlap/duration(nccl) if nccl else None,
        top_kernels=[dict(name=n,count=hot_count[n],total_us=t) for n,t in hot_time.most_common(12)],
        top_cpu_ops=cpu_names.most_common(12), top_cuda_api_us=api.most_common(10),
        grouped_api_audit=dict(calls=cpu_names['aten::_grouped_mm'],nested_cpu_ops=dict(grouped_children),
            scope='CPU operators nested under native grouped_mm API; an API name does not guarantee one fused GPU launch'),
        longest_observed_gaps=[dict(duration_us=d, start_us=t) for d,t in sorted(gaps,reverse=True)[:8]],
        layers=layer_rows, base_ns=reader.metadata.get('baseTimeNanoseconds'))
    row['unmatched_kernel_launches']=sorted(launches-{k[5] for k in kernels})
    row['message_gpu']=[dict(message=name,kernel_count=b['kernels'],global_layers=sorted(b['layers']),
        cpu_components=['stage%s.layer%s.%s' % label for label in sorted(cpu_components[name])],
        gpu_components=['stage%s.layer%s.%s' % label for label in sorted(gpu_components[name])],
        attention_union_us=duration(b['attention']),moe_union_us=duration(b['moe']),
        nccl_union_us=duration(b['nccl']),unassigned_union_us=duration(b['other'])) for name,b in by_message.items()]
    return row, dict(compute=union(compute),nccl=union(nccl))


def timeline(rows, tracks, path):
    """Binned overview only; exact CPU/stream/flow detail lives in full trace."""
    all_intervals=[x for values in tracks.values() for group in values.values() for x in group]
    if not all_intervals:
        path.write_text('<p>No GPU events; collection failed.</p>',encoding='utf-8')
        return
    origin=min(a for a,b in all_intervals)
    end=max(b for a,b in all_intervals)
    span=max(1,end-origin)
    svg=[]
    for index,row in enumerate(rows):
        y=28+index*46
        svg.append(f'<text x="0" y="{y}">rank{row["rank"]} stage{row["stages"]}</text>')
        for dy,kind,color in [(0,'compute','#2563eb'),(15,'nccl','#e87922')]:
            bins=set()
            for a,b in tracks[row['rank']][kind]:
                first=max(0,int((a-origin)/span*1100))
                last=min(1099,int((b-origin)/span*1100))
                bins.update(range(first,last+1))
            for value in sorted(bins):
                svg.append(f'<rect x="{210+value}" y="{y+dy-10}" width="1" height="11" fill="{color}"/>')
    path.write_text('<!doctype html><meta charset="utf-8"><title>多卡GPU轨迹概览</title>'
        '<h2>蓝：非NCCL kernel；橙：NCCL kernel</h2><p>每行一张卡，共同时间轴；1像素时间分桶，不能据像素判断微秒重叠。'
        'NCCL时长含等待，空白不等于PP气泡。精确stream、CPU与flow请打开trace.json.gz。</p>'
        f'<p>窗口 {span/1000:.3f} ms</p><svg viewBox="0 0 1340 {len(rows)*46+35}">'+''.join(svg)+'</svg>',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-ranks',type=int,required=True)
    parser.add_argument('--layers',type=int,default=48)
    parser.add_argument('--require-nccl',action='store_true')
    parser.add_argument('--same-host',action='store_true',help='assert one host, align differing Kineto base times')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows, tracks=[], {}
    for path in trace_paths(args.input_dir):
        print(f'Analyzing {path.name}',flush=True)
        row,track=analyze(path)
        rows.append(row)
        tracks[row['rank']]=track
    errors=[]
    if {r['rank'] for r in rows} != set(range(args.expected_ranks)):
        errors.append('Rank set does not match expected ranks')
    for row in rows:
        if (not all(row['categories'].get(k,0) for k in ('cpu_op','kernel','user_annotation'))
                or not (row['categories'].get('cuda_runtime') or row['categories'].get('cuda_driver'))
                or not row['layer_labels']):
            errors.append(f'rank{row["rank"]}: missing CPU/ATen/CUDA/layer activity')
        if args.require_nccl and not row['nccl_kernels']:
            errors.append(f'rank{row["rank"]}: no NCCL GPU kernels identified')
        from utils.trace_audit import coverage_errors
        manifest=Path(row['file']).parent/f'capture_rank{row["rank"]}.json'
        capture=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else None
        errors.extend(f'rank{row["rank"]}: {error}' for error in coverage_errors(row,capture))
    if set(l for r in rows for l in r['global_layers']) != set(range(args.layers)):
        errors.append('Global layer labels are incomplete')
    bases={r['base_ns'] for r in rows}
    aligned=len(bases)==1 and None not in bases
    if rows and None not in bases and args.same_host:
        common=min(bases)
        for row in rows:
            offset=(row['base_ns']-common)/1000
            tracks[row['rank']]={kind:[(a+offset,b+offset) for a,b in values]
                                 for kind,values in tracks[row['rank']].items()}
            row['alignment_offset_us']=offset
        aligned=True
    if not aligned:
        errors.append('Raw time bases differ or are absent; align traces before any cross-rank comparison')
    warnings=[]
    stage_compute=defaultdict(list)
    for row in rows:
        if len(row['stages'])==1:
            stage_compute[row['stages'][0]]+=tracks[row['rank']]['compute']
    pairs=[]
    if aligned:
        for a in sorted(stage_compute):
            for b in sorted(stage_compute):
                if a<b:
                    intersect=intersection(stage_compute[a],stage_compute[b])
                    combined=duration(stage_compute[a]+stage_compute[b])
                    pairs.append(dict(stages=[a,b],non_nccl_intersection_us=intersect,
                        intersection_over_union=intersect/combined if combined else None,
                        scope='all observed non-NCCL kernels, any rank in stage; not steady-state bubble or same-device communication overlap'))
    report=dict(collection_checks_passed=not errors,errors=errors,warnings=warnings,ranks=rows,stage_compute_overlap=pairs,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveats=['Message/layer/API coverage is checked; arbitrary event loss is not mathematically excluded.',
        'Layer GPU time uses CPU launch range -> correlation/External id attribution; unassigned kernels are excluded, not guessed.',
        'GPU interval unions count overlap once. CPU scopes and child kernels must never be summed together.',
        'Observed inactive fraction is not PP bubble or SM idle. NCCL duration includes waiting. No throughput from profiling.'])
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 多卡采集复核','',f'逐消息CPU/GPU层覆盖检查：{"通过" if not errors else "存在缺口或无法验证"}。这不等于数值或性能认证。','',
        '| Rank | Stage | ATen | CUDA kernel | NCCL kernel | GPU层归属比例 | 同卡计算通信交集 ms |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'|{r["rank"]}|{r["stages"]}|{r["categories"].get("cpu_op",0)}|{r["gpu_kernels"]}|{r["nccl_kernels"]}|{r["attribution_fraction"]:.1%}|{r["compute_nccl_overlap_us"]/1000:.3f}|')
    lines += ['','缺口：'+('; '.join(errors+warnings) or '基础事件检查未发现缺口；仍须逐消息检查尾层和profiler dropped记录。'),'',
        '逐层GPU时间、kernel热点、CPU操作数量、CUDA API耗时、最长可见空隙与每个消息名称见 summary.json。',
        '不要用NCCL kernel时长推算传输带宽；不要把同卡NCCL/计算交集写成跨stage流水线利用率。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if aligned:
        timeline(rows,tracks,args.output/'timeline.html')
    print(f'Saved {args.output}; collection checks: {not errors}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
