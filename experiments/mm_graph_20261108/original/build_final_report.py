"""Build a Chinese report from saved measurements, never from estimates."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=3):
    return f'{value:.{digits}f}' if value is not None else '—'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    root = args.evidence
    load = root/'load_matrix'
    suite = root/'completion_suite'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    rows = []
    all_rounds = []
    for path in sorted(load.glob('*/case.json')):
        if not path.parent.name.startswith(('offline_', 'online_')):
            continue
        state = read(path)
        rounds = read(path.parent/'results.json')
        first = rounds[0]
        cases.append((path.parent.name, state, rounds))
        samples = read(path.parent/'gpu_samples.json')
        peaks = {}
        util = []
        for sample in samples:
            if 'gpu' in sample:
                peaks[sample['gpu']] = max(peaks.get(sample['gpu'], 0), sample['memory_mib'])
                util.append(sample['utilization_pct'])
        for phase, selected in [('first', rounds[:1]), ('subsequent', rounds[1:])]:
            if not selected:
                continue
            requests = [q for r in selected for q in r['requests'] if 'error' not in q]
            prompts = [q['prompt_tokens'] for q in requests]
            row = dict(case=path.parent.name, backend=first['backend'], scope=first['scope'],
                concurrency=first['concurrency'], graph=first['graph'],
                max_pixels=first['max_pixels'], extra_context_target=first['context_tokens'],
                output_tokens_per_request=first['target_output_tokens'], phase=phase, rounds=len(selected),
                status=state['status'], completed=sum(r['completed'] for r in selected),
                errors=sum(r['errors'] for r in selected), prompt_min=min(prompts), prompt_max=max(prompts),
                output_tok_s=med([r['output_tok_s'] for r in selected]),
                pipeline_output_tok_s=med([r.get('pipeline_output_tok_s', r['output_tok_s']) for r in selected]),
                request_s=med([r['request_s'] for r in selected]),
                preprocessing_wall_s=med([r.get('preprocessing_wall_s') for r in selected]),
                offline_peak_active=max((r.get('peak_active',0) for r in selected),default=0) if first['backend']=='offline' else None,
                online_engine_states_peak_sampled=max((s['engine']['active'] for s in samples if 'engine' in s),default=None),
                nvml_peak_single_gpu_mib=max(peaks.values()) if peaks else None)
            for metric in ('ttft_s','e2e_s','itl_s','pipeline_ttft_s','pipeline_e2e_s'):
                for q in ('p50','p95','p99'):
                    row[f'{metric}_{q}'] = med([r.get(metric, {}).get(q) for r in selected])
            rows.append(row)
        for r in rounds:
            all_rounds.append(dict(case=path.parent.name,**{k:v for k,v in r.items() if k!='requests'}))
    assert len(cases)==26 and all(state['status']=='passed' and len(rs)==3 for _,state,rs in cases)
    graph_execution=[]
    folders=[load/name for name,_,_ in cases if name.startswith('offline_')] + [
        load/f'server_{s}_graph/worker' for s in ('text','multimodal')]
    for folder in folders:
        ranks=[read(p) for p in sorted(folder.glob('execution_rank*.json'))]
        assert len(ranks)==8, str(folder)
        signatures={(r['graph_requested'],r['graph_strict'],tuple(r['captured_buckets']),
            r['graph_capture_count'],r['graph_replays']) for r in ranks}
        assert len(signatures)==1, f'rank graph execution counts differ: {folder}'
        first=ranks[0]
        if first['graph_requested']:
            assert first['graph_strict'] and first['graph_replays']>0 and first['captured_buckets']
        graph_execution.append(dict(folder=str(folder.relative_to(root)),
            graph_requested=first['graph_requested'],graph_strict=first['graph_strict'],
            ranks=len(ranks),captured_buckets=first['captured_buckets'],
            captures_per_rank=first['graph_capture_count'],replays_per_rank=first['graph_replays'],
            pool_min_mib=min(r['graph_capture_bytes'] for r in ranks)/2**20,
            pool_max_mib=max(r['graph_capture_bytes'] for r in ranks)/2**20))
    (out/'graph_execution_summary.json').write_text(json.dumps(graph_execution,indent=2),encoding='utf-8')
    audit = read(root/'raw_output_audit.json')
    acceptance = read(suite/'acceptance.json')
    quality = read(suite/'quality_comparison.json')
    graph_eager = read(suite/'graph_eager_quality_comparison.json')
    vision = read(suite/'vision_real_256/vision.json')
    assert acceptance['passed'] and graph_eager['exact_token_matches']==256 and vision['passed']
    for name in ('strict_large','strict_large_release'):
        ranks=[read(p) for p in sorted((suite/name).glob('rank*.json'))]
        assert len(ranks)==8 and all(len(r['rows'])==56 and r['live_kv_unchanged'] and r['replays']>0 for r in ranks)
        for r in ranks:
            assert r['batch_invariant'] and r['strict_batch_invariant']
            for row in r['rows']:
                assert row['max_abs']==row['unpadded_max_abs']==0
                assert all(row[k] for k in ('exact_logits','tp_ranks_equal','kv_graph_equal',
                    'kv_padded_equal','permutation_equal','unpadded_argmax_equal'))
    source_sets = [state['source_sha256'] for _,state,_ in cases]
    assert all(s == source_sets[0] for s in source_sets), 'matrix source versions differ'
    total = sum(r['completed'] for _,_,rs in cases for r in rs)
    errors = sum(r['errors'] for _,_,rs in cases for r in rs)
    counts = {key:sum(c[key] for c in audit['checks']) for key in (
        'duplicate_input_compared','repeat_compared','sse_content_compared')}
    differences = {key:sum(len(c[key]) for c in audit['checks']) for key in (
        'duplicate_input_differences','repeat_differences','sse_content_differences','missing_token_evidence')}
    assert not any(differences.values()) and not audit['cross_case_differences'] and errors==0
    isolated=audit['isolated_comparisons']
    assert all(d['compared']>0 and not d['differences'] for d in isolated.values())
    with (out/'load_matrix_detailed.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (out/'load_matrix_detailed.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'load_matrix_all_rounds.json').write_text(json.dumps(all_rounds,indent=2),encoding='utf-8')
    lines = ['# CUDA Graph 与图像 Thinker：最终验证和负载报告', '',
        '实测日期：2026-11-08。环境：单机 8×NVIDIA A800-SXM4-80GB，Qwen3-Omni-30B-A3B-Instruct，BF16，TP8+EP8、DP1、PP1。',
        'PyTorch 2.14.1+cu130，CUDA 13.0，Transformers 5.19.0；vLLM 接入方式依据服务器安装的 0.32.0 源码核对。', '',
        '## 1. 已测结果和正确性边界', '',
        f'- 最终矩阵共 {len(cases)} 档、{sum(len(rs) for _,_,rs in cases)} 轮、{total:,} 个请求，错误 {errors}。其中 24 档为 Graph 在线/离线、文本/图像、128/256 并发、三档负载；另外两档为同策略 eager 对照。',
        f"- 同条件跨 case 的完整 token 序列对照 {audit['cross_case_compared']:,} 次，差异 0；重复输入对照 {counts['duplicate_input_compared']:,} 次、跨轮对照 {counts['repeat_compared']:,} 次，差异均为 0。计数表示比较次数，并非互不重叠的独立样本。",
        f"- 固定其它条件分别检查：跨并发 {isolated['concurrency']['compared']:,} 次、在线/离线 {isolated['backend']['compared']:,} 次、压力测试 Graph/eager {isolated['graph_eager']['compared']:,} 次，差异均为 0。不同类别可能使用同一请求，不能将这些计数相加当作独立样本数。",
        f"- 在线 SSE 可见文本与服务端原始 token 解码对照 {counts['sse_content_compared']:,} 次，差异 0；缺失 token 证据 0。",
        f"- 真实 ChartQA 256 题正常 EOS 停止：Graph/eager 完整 token 序列 {graph_eager['exact_token_matches']}/256 一致，使用相同 TP8+EP 和数值策略。",
        f"- 与 HF 单卡参考的完整 token 序列为 {quality['exact_token_matches']}/256 一致；HF relaxed accuracy {quality['hf_accuracy']*100:.6f}%，当前引擎 {quality['own_accuracy']*100:.6f}%。仍有 6 题差异，不能称为全量 HF 严格等价。",
        f"- 真实图像 encoder 对拍：{len(vision['rows'])} 个图像/分辨率组合，主 merger 和三路 DeepStack 均与实际 HF loader 逐位相同；未修改 HF rotary frequency buffer。",
        '- 严格 Graph 回归覆盖 batch 1/8/9/16/17/32/33/64/65/127/128/129/255/256；一组混合前缀为 17/255/256/511，另一组为 1023/1024/1535/2044，后者连续 decode 到 2048 的容量边界。两组各 4 步连续 decode、随机输入、重排、有效 KV 和 dummy KV 污染检查，每组 8 rank×56 个比较，完整 logits 最大误差 0。batch 与前缀按脚本混合分配，不是所有取值的笛卡尔积。',
        '- 传输修正后的最终 CPU 全套回归为 160 passed、19 skipped、3 subtests passed；跳过项不算 GPU 验证通过。更早的 transport 版本曾有一项纯 decode 不应传图片的回归失败，修正后的 core 已完整重测，失败日志也保留。', '',
        '这些结果覆盖本报告的有限输入和单机拓扑。音频、视频、Talker/Code2Wav、多机和其它并行拓扑尚未由本报告验证。', '',
        '## 2. 负载与计量方法', '',
        '数据为 HuggingFaceM4/ChartQA test 的固定 256 题子集（seed 20261108，完整 split 为 2,500 题）。质量测评使用真实图片、greedy、正常 EOS、最多 64 个输出 token；压力测试固定输出长度并忽略 EOS，其输出不能作为题目准确率。', '',
        '| 档位 | 最大图片像素 | 额外文本上下文目标 | 每请求输出 token |',
        '|---|---:|---:|---:|',
        '| 小 | 16,384 | 0 | 16 |', '| 中 | 65,536 | 256 | 64 |', '| 大 | 262,144 | 1,024 | 64 |', '',
        '额外上下文由重复句子构成，表内是构造参数，不是实际总 prompt token 数；实际最小/最大值在详细 CSV。文本 Thinker 请求包含相同问题和提示但不带图片，不能用文本与图像总耗时相减估计纯视觉耗时。', '',
        '每档闭环并发 C=128/256，总请求 2C，连续 3 轮；请求完成后补入。离线每个进程先做一个真实输入、2 个输出 token 的小预热。在线同一 scope 的各 case 复用一个服务；正式多模态服务的冒烟 case 被 harness 按已有结果复用，未在这个新服务进程中重新做图片预热，文本服务则执行自己的新冒烟 case。首轮可能含新 shape 编译、视觉首次执行、分配及缓存开销，因此首轮单独列出。后续栏为第 2、3 轮各项指标的中位数，p95 的中位数不是合并请求的 p95，也不支持统计显著性结论。', '',
        'C 表示客户端并发槽或离线活跃请求上限，不等于每次 GPU forward 的 batch。详细 CSV 另记离线实际 peak_active 和在线 health 中存活引擎请求状态数的采样峰值；后者含排队请求，不是 GPU 活跃 batch，且 1 秒采样可能漏过瞬时峰值。在线图像预处理最多 4 个并行槽，GPU admission 也受到每步 2,048 token 预算限制。', '',
        'offline engine window 包括视觉编码（图像模式）、Thinker prefill/decode、调度和 worker IPC；pipeline 再包含读图、解码、patch、模板和 tokenization。offline TTFT/E2E 从请求进入引擎开始；pipeline 延迟从整批输入预处理开始的共同起点计算，包含尚未获得并发槽的等待。online 从请求获得客户端并发槽开始，包含读图/base64、HTTP、服务端预处理、排队、GPU 计算和 SSE；客户端和服务端同机通过 loopback 通信。两种 TTFT 的起点不同。', '',
        'ITL 在线按可见内容片段间隔计算，离线按引擎 token 事件计算。NVML 为每档运行期采样的最大单卡使用量，包含 context 和 allocator 缓存；不是仅权重或活跃 tensor 大小。初始化和权重装载不在吞吐窗口内。', '',
        '## 3. 总表：Graph 的后续轮次', '',
        '| 模式 | 路径 | 档位 | 并发 | window 输出 tok/s | pipeline 输出 tok/s | TTFT p95 s | E2E p95 s | ITL p95 ms | 单卡峰值 MiB |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    mode = {'offline':'离线','online':'在线 HTTP'}
    scope = {'text':'Thinker 文本','multimodal':'完整图像链路'}
    size = {16384:'小',65536:'中',262144:'大'}
    for r in rows:
        if r['phase']=='subsequent' and r['graph']:
            lines.append('| '+' | '.join([mode[r['backend']],scope[r['scope']],size[r['max_pixels']],str(r['concurrency']),
                fmt(r['output_tok_s']),fmt(r['pipeline_output_tok_s']),fmt(r['ttft_s_p95']),fmt(r['e2e_s_p95']),
                fmt(r['itl_s_p95']*1000 if r['itl_s_p95'] is not None else None),str(r['nvml_peak_single_gpu_mib'])])+' |')
    lines += ['', '## 4. 首轮吞吐与 Graph/eager 对照', '',
        '| Case | 首轮 window tok/s | 首轮 pipeline tok/s | 后续 window tok/s | 后续 pipeline tok/s |',
        '|---|---:|---:|---:|---:|']
    lookup={(r['case'],r['phase']):r for r in rows}
    for name,_,_ in cases:
        first=lookup[(name,'first')];sub=lookup[(name,'subsequent')]
        lines.append(f"| {name} | {fmt(first['output_tok_s'])} | {fmt(first['pipeline_output_tok_s'])} | {fmt(sub['output_tok_s'])} | {fmt(sub['pipeline_output_tok_s'])} |")
    lines += ['', '以下比较只针对中档离线图像负载，保持相同固定数值策略和 dense EP backend；不能与关闭严格数值策略的普通 eager 混比。顺序运行且只有两轮后续观测，变化为描述性结果。', '',
        '| 并发 | eager window tok/s | Graph window tok/s | 变化 | eager pipeline tok/s | Graph pipeline tok/s | 变化 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for c in (128,256):
        e=lookup[(f'offline_multimodal_medium_c{c}_eager','subsequent')]
        g=lookup[(f'offline_multimodal_medium_c{c}_graph','subsequent')]
        lines.append(f"| {c} | {fmt(e['output_tok_s'])} | {fmt(g['output_tok_s'])} | {(g['output_tok_s']/e['output_tok_s']-1)*100:+.2f}% | {fmt(e['pipeline_output_tok_s'])} | {fmt(g['pipeline_output_tok_s'])} | {(g['pipeline_output_tok_s']/e['pipeline_output_tok_s']-1)*100:+.2f}% |")
    lines += ['', '## 5. HF 差异逐题保留', '',
        '| 原始 row | HF | 当前引擎 | HF 正确 | 当前正确 |', '|---:|---|---|---|---|']
    for d in quality['disagreements']:
        lines.append(f"| {d['index']} | {d['hf']} | {d['own']} | {d['hf_correct']} | {d['own_correct']} |")
    lines += ['', '256/256 的 Graph/eager 生成一致性表明，这些 HF 差异在本轮条件下并非开启 Graph 后新增。该结论不能推出 TP8 logits 与 HF 单卡 logits 逐位一致；row 447 的强制前缀诊断已有非零语言 logits 差异。', '',
        '## 6. 代码与复现证据', '',
        '所有矩阵 case 保存相同完整 v9 Python 源码 SHA256 清单、命令、PID、起止、逐请求 token、3 轮结果及显存采样。每个离线进程和两个在线服务均记录 8 个 rank 的 captured_buckets、graph_replays、graph_strict；Graph 模式实际 replay 数均大于 0。', '',
        '- `load_matrix_detailed.csv`：52 行首轮/后续阶段数据，包含 p50/p95/p99、实际 prompt 长度范围、吞吐及显存。',
        '- `load_matrix_all_rounds.json`：全部 78 轮原始聚合指标，首轮没有被丢弃。',
        '- `graph_execution_summary.json`：每个离线进程及在线服务的 8 rank 捕获/回放次数、bucket 和 Graph pool；同一进程组的 rank 计数已核对一致。',
        '- `final_benchmark_evidence/`：原始负载、HTTP token、执行元数据、质量、encoder 对拍、Graph 回归与审计。',
        '- `cuda_graph_and_multimodal_explainer.md`：数值策略、每 GPU rank 的捕获协议、图像输入/Thinker/Talker 边界、vLLM 接入对照及修复原因。', '',
        '服务器工作目录：`/data/ifly_jyhe23/vllm-learn-mm-benchmark-v2-20261108`。先执行 `run_completion_suite.py`，完成后执行 `finalize_results.py`；运行环境为 `/home/ifly_jyhe23/env/miniconda3/envs/sglang_contrib/bin/python`。原有结果目录有保护，重复实验应使用独立目录，避免覆盖证据。', '']
    (out/'cuda_graph_multimodal_final_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('REPORT_WRITTEN',len(rows),'phase rows',total,'requests',errors,'errors')


if __name__=='__main__':
    main()
