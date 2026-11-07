"""Summarize completed load runs without conflating throughput and correctness."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def load(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def summarize(root):
    rows = []
    consistency = []
    baselines = {}
    for path in sorted(root.glob('*/results.json')):
        case = path.parent
        if not case.name.startswith(('online_', 'offline_')):
            continue
        rounds = load(path)
        if not rounds:
            continue
        first = rounds[0]
        server = root/f"server_{first['scope']}_graph"
        http = {}
        if first['backend'] == 'online' and (server/'http_requests.jsonl').exists():
            for line in (server/'http_requests.jsonl').read_text().splitlines():
                record = json.loads(line)
                http[record['id']] = record
        memory_root = case if first['backend'] == 'offline' else server/'worker'
        memory = [load(f) for f in memory_root.glob('memory_rank*.json')]
        samples = load(case/'gpu_samples.json', [])
        per_gpu = {}
        for sample in samples:
            if 'gpu' in sample:
                key = sample['gpu']
                per_gpu[key] = max(per_gpu.get(key, 0), sample['memory_mib'])
        refs = {}
        group = (first['scope'], first['backend'], first['concurrency'],
                 first['max_pixels'], first['context_tokens'], first['target_output_tokens'])
        for repeat in rounds:
            seen = {}
            duplicate = []
            changed = []
            missing = []
            cross_graph = []
            for request in repeat['requests']:
                if 'error' in request:
                    continue
                ordinal = request.get('request_index')
                if 'output_ids' in request:
                    ids = request['output_ids']
                else:
                    ids = http.get(request.get('request_id'), {}).get('output_ids')
                if ids is None:
                    missing.append(ordinal)
                    continue
                index = request['index']
                if index in seen and seen[index] != ids:
                    duplicate.append(dict(request_index=ordinal, index=index))
                seen.setdefault(index, ids)
                if ordinal is not None:
                    if ordinal in refs and refs[ordinal] != ids:
                        changed.append(ordinal)
                    refs.setdefault(ordinal, ids)
                    key = (group, ordinal)
                    if key in baselines and baselines[key][0] != first['graph']:
                        if baselines[key][1] != ids:
                            cross_graph.append(ordinal)
                    baselines.setdefault(key, (first['graph'], ids))
            consistency.append(dict(case=case.name, repeat=repeat['repeat'],
                duplicate_input_differences=duplicate, repeat_differences=changed,
                graph_eager_differences=cross_graph, missing_token_evidence=missing))
        # The first measured round follows only a tiny real-image warmup;
        # never silently exclude its compilation and allocator costs.
        for phase, selected in [('first', rounds[:1]), ('subsequent', rounds[1:])]:
            if not selected:
                continue
            med = lambda key: statistics.median(r[key] for r in selected)
            metric = lambda key: statistics.median(r[key]['p95'] for r in selected
                if r[key]['p95'] is not None) if any(r[key]['p95'] is not None for r in selected) else None
            prompts = [q['prompt_tokens'] for r in selected for q in r['requests'] if 'prompt_tokens' in q]
            rows.append(dict(case=case.name, backend=first['backend'], scope=first['scope'],
                graph=first['graph'], concurrency=first['concurrency'], max_pixels=first['max_pixels'],
                extra_context_target=first['context_tokens'], output_length=first['target_output_tokens'],
                phase=phase, rounds=len(selected), completed=sum(r['completed'] for r in selected),
                errors=sum(r['errors'] for r in selected), prompt_min=min(prompts) if prompts else None,
                prompt_max=max(prompts) if prompts else None,
                window_output_tok_s=med('output_tok_s'),
                pipeline_output_tok_s=med('pipeline_output_tok_s') if 'pipeline_output_tok_s' in first else med('output_tok_s'),
                ttft_p95_s=metric('ttft_s'), e2e_p95_s=metric('e2e_s'), itl_p95_s=metric('itl_s'),
                pipeline_ttft_p95_s=metric('pipeline_ttft_s') if 'pipeline_ttft_s' in first else metric('ttft_s'),
                preprocessing_wall_s=med('preprocessing_wall_s') if 'preprocessing_wall_s' in first else None,
                nvml_peak_gpu_mib=max(per_gpu.values()) if per_gpu else None,
                nvml_sum_individual_gpu_peaks_mib=sum(per_gpu.values()) if per_gpu else None,
                worker_max_peak_allocated_gib=max(m['peak_allocated_bytes'] for m in memory)/2**30 if memory else None,
                memory_scope='case lifetime' if first['backend']=='offline' else 'NVML: case; allocator: shared server lifetime',
                completed_rounds=len(rounds), status=load(case/'case.json',{}).get('status','unknown')))
    return rows, consistency


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, consistency = summarize(args.root)
    (args.output/'matrix.json').write_text(json.dumps(rows, indent=2))
    (args.output/'consistency.json').write_text(json.dumps(consistency, indent=2))
    if rows:
        with (args.output/'matrix.csv').open('w',newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ['# 128/256 并发评测', '',
        '每一行是一档负载的一个阶段；后续轮数字为各轮指标的中位数，p95 的中位数不是合并请求后的 p95。',
        '离线 window 包括调度和 Thinker，含图像时也包括视觉编码；pipeline 再包括输入预处理。在线均为 HTTP 完整链路。',
        'first 为首个测量轮，subsequent 为其后全部测量轮；启动装载不包含在吞吐窗口中。',
        'NVML 统计包含 CUDA context/缓存等；各 GPU 峰值求和不是同一时刻的整机峰值。在线 worker allocator 峰值跨该服务的所有 case。',
        'case 的 passed 只表示运行完成；逐 token 一致性见 consistency.json。', '',
        '| Case | 阶段 | 请求/错误 | window tok/s | pipeline tok/s | TTFT p95(s) | E2E p95(s) | NVML 最大单卡 MiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        fmt = lambda x: f'{x:.3f}' if isinstance(x,float) else str(x) if x is not None else '—'
        lines.append('| '+' | '.join([row['case'],row['phase'],f"{row['completed']}/{row['errors']}",
            fmt(row['window_output_tok_s']),fmt(row['pipeline_output_tok_s']),fmt(row['ttft_p95_s']),
            fmt(row['e2e_p95_s']),fmt(row['nvml_peak_gpu_mib'])])+' |')
    (args.output/'matrix.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('SUMMARIZED', len(rows), 'phase rows;', len(consistency), 'consistency checks')


if __name__ == '__main__':
    main()
