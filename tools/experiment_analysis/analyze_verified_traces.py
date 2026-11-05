"""Create compact, labeled Chrome traces and quantitative summaries."""
from collections import Counter, defaultdict
from bisect import bisect_right
import json
import math
from pathlib import Path
import re
import statistics
from html import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "verified_parallel"
LAYER = re.compile(r"stage(\d+)\.layer\.?([0-9]+)\.(attention|moe)$")
MICRO = re.compile(r"stage(\d+)\.microbatch(\d+)\.lane(\d+)\.(prefill|decode)")


def interval_union(intervals):
    total = 0.0
    end = -math.inf
    for a, b in sorted(intervals):
        total += max(0, b-max(a, end))
        end = max(end, b)
    return total


def merged_intervals(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a,b))
    return result


def interval_intersection(a, b):
    a, b = merged_intervals(a), merged_intervals(b)
    i = j = 0
    total = 0.
    while i < len(a) and j < len(b):
        total += max(0., min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1] <= b[j][1]: i += 1
        else: j += 1
    return total


def enclosing(ranges, starts, event):
    index = bisect_right(starts, event['ts'])-1
    if index >= 0:
        candidate = ranges[index]
        if (candidate.get('tid') == event.get('tid') and
                event['ts'] < candidate['ts']+candidate.get('dur', 0)):
            return candidate
    return None


def load_chrome(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, data.get("traceEvents", [])


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def summarize_run(name, ranks, pp_size, ep_size, prefix="cuda"):
    out_dir = OUT / name
    raw_dir = out_dir / "trace"
    events_out = []
    rows = defaultdict(list)
    rank_info = []
    kernel_names = Counter()
    annotation_totals = defaultdict(float)
    stream_names = {}

    for rank in range(ranks):
        path = raw_dir / f"{prefix}_rank{rank}.json"
        if not path.exists():
            continue
        data, events = load_chrome(path)
        pp_rank = rank // (ep_size if pp_size > 1 else 8)
        ep_rank = rank % ep_size if ep_size > 1 else rank
        props = data.get("deviceProperties", [])
        gpu_name = props[rank].get("name", "GPU") if rank < len(props) else "GPU"
        timed = [e for e in events if e.get("ph") == "X" and "ts" in e]
        rank_info.append({
            "rank": rank, "gpu": rank, "gpu_name": gpu_name,
            "pp_stage": pp_rank, "ep_rank": ep_rank,
            "base_time_ns": data.get("baseTimeNanoseconds"),
            "event_start_us": min((e["ts"] for e in timed), default=None),
            "event_end_us": max((e["ts"] + e.get("dur", 0) for e in timed), default=None),
            "event_count": len(events),
            "categories": dict(Counter(e.get("cat", "unknown") for e in events)),
        })
        stream_names[rank] = set()
        host_layers = sorted((e for e in events if e.get('cat') == 'user_annotation'
                              and e.get('ph') == 'X' and LAYER.search(e.get('name', ''))),
                             key=lambda e: e['ts'])
        host_starts = [e['ts'] for e in host_layers]
        host_micro = sorted((e for e in events if e.get('cat') == 'user_annotation'
                             and e.get('ph') == 'X' and MICRO.search(e.get('name',''))),
                            key=lambda e:e['ts'])
        micro_starts = [e['ts'] for e in host_micro]
        external_to_layer = {}
        external_to_micro = {}
        for host_op in events:
            if host_op.get('cat') not in ('cpu_op', 'cuda_runtime', 'user_annotation'):
                continue
            if 'ts' not in host_op or 'External id' not in host_op.get('args', {}):
                continue
            host_range = enclosing(host_layers, host_starts, host_op)
            if host_range:
                external_to_layer[host_op['args']['External id']] = LAYER.search(host_range['name']).groups()
            mb_range = enclosing(host_micro, micro_starts, host_op)
            if mb_range:
                external_to_micro[host_op['args']['External id']] = MICRO.search(mb_range['name']).groups()

        for event in events:
            cat = event.get("cat")
            if cat not in {"kernel", "gpu_user_annotation", "gpu_memcpy", "gpu_memset"}:
                continue
            if event.get("ph") != "X":
                continue
            item = dict(event)
            args = dict(item.get("args", {}))
            stream = args.get("stream", event.get("tid", "unknown"))
            device = args.get("device", rank)
            item["pid"] = rank
            item["tid"] = int(stream) if str(stream).isdigit() else str(stream)
            args.update(rank=rank, gpu=rank, pp_stage=pp_rank,
                        ep_rank=ep_rank, device=device, raw_category=cat)
            micro = external_to_micro.get(args.get('External id'))
            if micro:
                stage, mb, lane, phase = micro
                args.update(microbatch_id=int(mb), lane=int(lane), phase=phase)
            name_match = LAYER.search(event.get("name", ""))
            if name_match:
                stage, layer, op = name_match.groups()
                layer, stage = int(layer), int(stage)
                args.update(stage=stage, layer_global=layer,
                            op_type="attention" if op == "attention" else "moe")
                item["name"] = f"L{layer:02d} {'注意力' if op == 'attention' else 'MoE'}"
                if cat == "gpu_user_annotation":
                    rows[(stage, layer, op, rank)].append(float(event.get("dur", 0)))
            elif "nccl:" in event.get("name", ""):
                args["op_type"] = "NCCL"
                item["name"] = event["name"].replace("nccl:", "通信:")
                if cat == "gpu_user_annotation":
                    annotation_totals[(rank, "NCCL")] += float(event.get("dur", 0))
            elif cat == "kernel":
                kernel_names[event.get("name", "kernel")] += 1
                inherited = external_to_layer.get(args.get('External id'))
                if inherited:
                    stage, layer, op = inherited
                    args.update(stage=int(stage), layer_global=int(layer), op_type=op,
                                layer_attribution='CPU External id -> enclosing layer launch range',
                                original_kernel_name=event.get('name'))
                    item['name'] = f'L{int(layer):02d} {"注意力" if op == "attention" else "MoE"} | ' + event.get('name','kernel')
            item["args"] = args
            if micro and ('layer_global' in args or cat == 'gpu_user_annotation'):
                item['name'] = f"MB{args['microbatch_id']}/{args['phase']} | " + item['name']
            events_out.append(item)
            stream_names[rank].add(str(stream))

        rank_gpu = [e for e in events_out if e['pid'] == rank and e['cat'] in ('kernel','gpu_memcpy','gpu_memset')]
        kernels = [e for e in rank_gpu if e['cat'] == 'kernel']
        compute = [e for e in kernels if 'nccl' not in e['name'].lower()]
        comm = [e for e in kernels if 'nccl' in e['name'].lower()]
        span = (max(e['ts']+e.get('dur',0) for e in rank_gpu)-min(e['ts'] for e in rank_gpu)) if rank_gpu else 0
        intervals = lambda xs:[(e['ts'],e['ts']+e.get('dur',0)) for e in xs]
        cu, nu = interval_union(intervals(compute)), interval_union(intervals(comm))
        union = interval_union(intervals(compute+comm))
        rank_info[-1].update(
            kernel_count=len(kernels),
            kernels_with_layer=sum('layer_global' in e['args'] for e in kernels),
            kernels_with_microbatch=sum('microbatch_id' in e['args'] for e in kernels),
            gpu_window_ms=span/1000,
            non_nccl_kernel_union_ms=cu/1000,
            nccl_kernel_union_ms=nu/1000,
            compute_nccl_overlap_ms=max(0, cu+nu-union)/1000,
            kernel_union_fraction=union/span if span else None,
            sampled_microbatches=sorted({(e['args']['phase'], e['args']['microbatch_id']) for e in rank_gpu if 'microbatch_id' in e['args']}),
            metric_caveat='Interval union in a short profiled window, not SM utilization; NCCL kernels may spin or wait. No full-request bubble ratio inferred.')
        expected_layers = set(range(48*pp_rank//pp_size, 48*(pp_rank+1)//pp_size))
        kernel_layers = {e['args']['layer_global'] for e in kernels if 'layer_global' in e['args']}
        annotated_layers = {int(LAYER.search(e['name']).group(2)) for e in host_layers}
        rank_info[-1].update(
            expected_global_layers=sorted(expected_layers),
            host_annotated_global_layers=sorted(annotated_layers),
            kernel_attributed_global_layers=sorted(kernel_layers),
            missing_kernel_attributed_layers=sorted(expected_layers-kernel_layers),
            complete_layer_kernel_coverage=(kernel_layers >= expected_layers))
        capture_path = raw_dir / f'capture_rank{rank}.json'
        if capture_path.exists():
            capture = json.loads(capture_path.read_text(encoding='utf-8'))
            coverage = []
            for message in capture.get('captured_messages', []):
                phase = 'prefill' if message['is_prefill'] else 'decode'
                mb = message['microbatch_id']
                matching = [e for e in kernels if e['args'].get('microbatch_id') == mb
                            and e['args'].get('phase') == phase]
                layers = {e['args']['layer_global'] for e in matching if 'layer_global' in e['args']}
                coverage.append(dict(message_index=message['message_index'], microbatch_id=mb,
                    phase=phase, kernel_count=len(matching), missing_layers=sorted(expected_layers-layers),
                    complete=layers >= expected_layers))
            rank_info[-1]['per_message_kernel_coverage'] = coverage
            rank_info[-1]['complete_all_captured_messages'] = bool(coverage) and all(c['complete'] for c in coverage)

    # Explicit labels make rank, device, and CUDA stream visible in Chrome.
    labels = []
    for info in rank_info:
        rank = info["rank"]
        labels.append({"name": "process_name", "ph": "M", "pid": rank,
                       "args": {"name": f"GPU {rank} / rank {rank} / PP stage {info['pp_stage']} / EP rank {info['ep_rank']}"}})
        for stream in sorted(stream_names.get(rank, set())):
            tid = int(stream) if stream.isdigit() else stream
            labels.append({"name": "thread_name", "ph": "M", "pid": rank,
                           "tid": tid, "args": {"name": f"CUDA stream {stream} | GPU {rank}"}})

    layer_rows = []
    by_layer = defaultdict(lambda: {"attention_ms": [], "moe_ms": []})
    for (stage, layer, op, rank), values in rows.items():
        duration = mean(values) / 1000.0
        by_layer[(stage, layer)][f"{op}_ms"].append(duration)
        layer_rows.append({"stage": stage, "layer_global": layer,
                           "rank": rank, "gpu": rank, "op": op,
                           "samples": len(values), "mean_gpu_range_ms": duration,
                           "p95_gpu_range_ms": (sorted(values)[math.ceil(.95*len(values))-1]/1000.0 if values else 0)})
    layer_summary = []
    for (stage, layer), ops in sorted(by_layer.items()):
        attn, moe = mean(ops["attention_ms"]), mean(ops["moe_ms"])
        layer_summary.append({"stage": stage, "layer_global": layer,
                              "attention_mean_gpu_range_ms": attn,
                              "moe_mean_gpu_range_ms": moe,
                              "moe_to_attention_ratio": moe / attn if attn else None})

    summary = {
        "name": name, "pp_size": pp_size, "ep_size": ep_size,
        "trace_scope": "CUDA profiler GPU events from the first active profile window; not the throughput run",
        "included_categories": ["kernel", "gpu_user_annotation", "gpu_memcpy", "gpu_memset"],
        "excluded_categories": ["cpu_op", "cuda_runtime", "user_annotation"],
        "clock_note": "Raw CUDA profiler timestamps retained. Compare ranks only after checking baseTimeNanoseconds and overlapping start/end ranges.",
        "ranks": rank_info,
        "layer_gpu_ranges": layer_rows,
        "layer_summary": layer_summary,
        "nccl_gpu_range_sum_ms_by_rank": {
            str(rank): ms / 1000.0 for (rank, _), ms in annotation_totals.items()},
        "top_cuda_kernel_names": kernel_names.most_common(20),
        "merged_event_count": len(events_out),
    }
    bases = {r['base_time_ns'] for r in rank_info}
    if len(bases) != 1 or None in bases:
        raise ValueError(f'{name}: rank time origins are incompatible: {bases}')
    # Pipeline overlap is cross-stage; same-device compute/NCCL overlap alone
    # cannot tell whether two different stages ran useful kernels together.
    compute_by_stage = defaultdict(list)
    for event in events_out:
        if event.get('cat') == 'kernel' and 'nccl' not in event['name'].lower():
            compute_by_stage[event['args']['pp_stage']].append((event['ts'],event['ts']+event.get('dur',0)))
    stage_overlap = []
    for a in range(pp_size):
        for b in range(a+1,pp_size):
            ai, bi = compute_by_stage[a], compute_by_stage[b]
            au, bu = interval_union(ai), interval_union(bi)
            overlap = interval_intersection(ai,bi)
            stage_overlap.append(dict(stage_a=a,stage_b=b,compute_union_a_ms=au/1000,
                compute_union_b_ms=bu/1000,compute_concurrent_ms=overlap/1000,
                concurrent_fraction_of_smaller=overlap/min(au,bu) if min(au,bu) else None))
    summary['cross_stage_compute_overlap'] = stage_overlap
    # Restrict stronger pipeline claims to messages whose layer-attributed
    # kernels cover every expected layer on every rank. Retain incomplete
    # windows in the artifact, but do not interpret missing tail data as idle.
    complete_sets = []
    for info in rank_info:
        complete_sets.append({(c['phase'], c['microbatch_id'])
            for c in info.get('per_message_kernel_coverage', []) if c['complete']})
    common = set.intersection(*complete_sets) if len(complete_sets) == ranks else set()
    summary['common_complete_messages'] = [dict(phase=p, microbatch_id=m) for p,m in sorted(common)]
    complete_stage_intervals = defaultdict(list)
    for event in events_out:
        a = event['args']
        if (event.get('cat') == 'kernel' and 'nccl' not in event['name'].lower()
                and (a.get('phase'), a.get('microbatch_id')) in common):
            complete_stage_intervals[a['pp_stage']].append((event['ts'], event['ts']+event.get('dur',0)))
    complete_overlap = []
    for a in range(pp_size):
        for b in range(a+1,pp_size):
            ai, bi = complete_stage_intervals[a], complete_stage_intervals[b]
            au, bu = interval_union(ai), interval_union(bi)
            overlap = interval_intersection(ai,bi)
            complete_overlap.append(dict(stage_a=a, stage_b=b, compute_union_a_ms=au/1000,
                compute_union_b_ms=bu/1000, compute_concurrent_ms=overlap/1000,
                concurrent_fraction_of_smaller=overlap/min(au,bu) if min(au,bu) else None))
    summary['complete_message_cross_stage_compute_overlap'] = complete_overlap
    summary['complete_message_scope'] = ('Only messages with all expected layers attributed on all ranks; '
        'layer coverage does not prove every kernel was captured. Excluding edge messages also '
        'excludes their real overlap; these are selected-window observations, not steady-state utilization.')
    summary['cross_stage_overlap_caveat'] = ('Union of any non-NCCL kernel in each stage, '
        'not useful FLOPs or SM utilization; capture windows can be offset, and '
        'warmup/fill/drain must be distinguished from steady state.')
    events_out.sort(key=lambda e: e.get('ts', 0))
    merged = {"displayTimeUnit": "ms", "traceEvents": labels + events_out,
              "metadata": {"model": "Qwen3-Omni-30B-A3B thinker text path",
                           "parallel": name, "rank_mapping": "rank = PP-major then EP; one TP rank",
                           "tracks": "process=GPU/rank/stage; thread=CUDA stream",
                           "baseTimeNanoseconds": next(iter(bases)),
                           "attribution": "CPU External id and enclosing launch ranges; unassigned kernels retain original names",
                           "only_gpu_events": True}}
    (OUT / f"{name}_gpu_timeline.json").write_text(
        json.dumps(merged, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT / f"{name}_trace_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_svg(summary, output):
    events = json.loads((OUT / f"{summary['name']}_gpu_timeline.json").read_text(encoding="utf-8"))["traceEvents"]
    events = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not events:
        return
    t0 = min(e["ts"] for e in events)
    t1 = max(e["ts"] + e.get("dur", 0) for e in events)
    width, left, right, row_h, top = 1500, 240, 30, 28, 70
    scale = (width-left-right) / max(t1-t0, 1)
    tracks = sorted({(e['pid'],str(e['tid'])) for e in events})
    rank_stages = {r['rank']: r['pp_stage'] for r in summary['ranks']}
    height = top + row_h * len(tracks) + 70
    colors = {"attention": "#2A9D8F", "moe": "#E76F51", "NCCL": "#457B9D",
              "other": "#64748B", "copy": "#E9C46A"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="{left}" y="28" font-size="20" font-family="sans-serif">{summary["name"]}：CUDA kernels / GPU / stream 时间线</text>']
    for i, (rank, stream) in enumerate(tracks):
        y = top + i*row_h
        lines.append(f'<text x="8" y="{y+17}" font-size="12" font-family="sans-serif">GPU{rank} / stage{rank_stages[rank]} / stream{stream}</text>')
        lines.append(f'<line x1="{left}" y1="{y+row_h}" x2="{width-right}" y2="{y+row_h}" stroke="#e5e7eb"/>')
    for event in events:
        rank = event["pid"]
        cat = ("copy" if event["cat"] != "kernel" else
               ("NCCL" if "nccl" in event["name"].lower() else event["args"].get("op_type", "other")))
        if cat not in colors:
            continue
        x = left + (event["ts"]-t0)*scale
        y = top + tracks.index((rank,str(event['tid'])))*row_h + 4
        w = max(1.0, event.get("dur", 0)*scale)
        tooltip = escape(event['name'] + ' | ' + str(round(event.get('dur',0),1)) + ' us', quote=True)
        lines.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="17" rx="1" fill="{colors[cat]}" opacity="0.88"><title>{tooltip}</title></rect>')
    legend_y = height-25
    for frac in (0,.25,.5,.75,1):
        x=left+(width-left-right)*frac
        lines.append(f'<text x="{x}" y="{top-9}" font-size="11">{(t1-t0)*frac/1000:.1f} ms</text>')
    for i,(label,color) in enumerate((("注意力层kernel",colors["attention"]),("MoE层kernel",colors["moe"]),("NCCL",colors["NCCL"]),("未归属kernel",colors["other"]),("memcpy/memset",colors["copy"]))):
        x=left+i*110
        lines.append(f'<rect x="{x}" y="{legend_y-12}" width="12" height="12" fill="{color}"/><text x="{x+18}" y="{legend_y}" font-size="12" font-family="sans-serif">{label}</text>')
    lines.append('<text x="8" y="%s" font-size="11" font-family="sans-serif">CUDA kernel events按实际stream分行；有CUDA活动的长区间不等于SM有效利用率。用JSON查看原始kernel和层归属。</text>' % (height-5))
    lines.append('</svg>')
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overnight', action='store_true')
    options = parser.parse_args()
    summaries = []
    runs = (("ep8_trace", 1, 8), ("pp2ep4_trace", 2, 4),
                         ('pp_prefill_baseline', 2, 4), ('pp_prefill_optimized', 2, 4),
                         ('ep8_expert_profile', 1, 8))
    prefix = 'cuda'
    if options.overnight:
        runs = tuple(('overnight/'+name, pp, ep) for name,pp,ep in (
            ('ep8_window',1,8), ('pp_baseline_window',2,4),
            ('pp_optimized_window',2,4), ('ep8_ownership_window',1,8),
            ('ep8_4k_window',1,8), ('pp2ep4_4k_window',2,4),
            ('pp_dedicated_window',2,4), ('ep8_warm_window',1,8),
            ('pp_dedicated_warm_window',2,4), ('reference_ep8_window',1,8),
            ('reference_pp_baseline_window',2,4), ('reference_pp_optimized_window',2,4)))
        # capture_rankN.json describes the window; CUDA events remain in cuda_rankN.json.
        prefix = 'cuda'
    for name, pp, ep in runs:
        trace_dir = OUT / name / 'trace'
        expected = [trace_dir / f'{prefix}_rank{rank}.json' for rank in range(8)]
        if not all(path.exists() for path in expected):
            print(f'Skip {name}: expected CUDA traces for ranks 0-7')
            continue
        if any(trace_dir.glob('*.partial')):
            print(f'Skip {name}: partial trace files are present')
            continue
        summaries.append(summarize_run(name, 8, pp, ep, prefix=prefix))
        write_svg(summaries[-1], OUT / f"{name}_gpu_timeline.svg")
    (OUT / ("overnight/combined_trace_analysis.json" if options.overnight else "combined_trace_analysis.json")).write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote merged GPU traces and rank/layer summaries")
