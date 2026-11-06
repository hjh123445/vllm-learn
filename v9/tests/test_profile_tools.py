"""CPU-only contracts for streaming IO, GPU attribution, and real interval overlap."""
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.trace_io import TraceReader, trace_paths
from tools.profile_analyze import analyze
from tools.profile_merge import merge_traces


def test_streaming_reader_preserves_before_after_metadata_and_large_events(tmp_path):
    events=[dict(name='中文'+('x'*100000),ph='X',ts=12,dur=3)]
    data=dict(baseTimeNanoseconds=123,traceEvents=events,metadata=dict(run='same'))
    path=tmp_path/'cuda_rank0.json.gz'
    with gzip.open(path,'wt',encoding='utf8') as handle:
        json.dump(data,handle,ensure_ascii=False)
    reader=TraceReader(path)
    assert list(reader.events())==events
    assert reader.metadata==dict(baseTimeNanoseconds=123,metadata=dict(run='same'))


def test_analyzer_uses_correlation_and_intervals_instead_of_nested_sums(tmp_path):
    def event(name,cat,ts,dur,pid=123,tid=2,**args):
        return dict(name=name,cat=cat,ph='X',ts=ts,dur=dur,pid=pid,tid=tid,args=args)
    events=[event('stage0.layer0.moe','user_annotation',90,40),
        event('aten::mm','cpu_op',100,2,**{'External id':42}),
        event('cudaLaunchKernel','cuda_runtime',100,1,**{'External id':42,'correlation':7}),
        event('gemm','kernel',110,10,pid=0,tid=7,correlation=7),
        event('ncclAllToAll','kernel',115,10,pid=0,tid=9),
        event('gpu_range','gpu_user_annotation',100,10000,pid=0,tid=7)]
    path=tmp_path/'cuda_rank0.json'
    path.write_text(json.dumps(dict(traceEvents=events,baseTimeNanoseconds=1)))
    row,_=analyze(path)
    assert row['compute_nccl_overlap_us']==5
    assert row['gpu_activity_union_us']==15
    assert row['layers'][0]['gpu_kernel_union_us']==10
    assert row['attributed_kernels']==1


def test_driver_api_launches_attribute_triton_to_layer(tmp_path):
    path=tmp_path/'cuda_rank0.json'
    path.write_text(json.dumps(dict(baseTimeNanoseconds=1,traceEvents=[
        dict(name='stage0.layer0.moe',cat='user_annotation',ph='X',pid=2,tid=3,ts=10,dur=10),
        dict(name='cuLaunchKernelEx',cat='cuda_driver',ph='X',pid=2,tid=3,ts=12,dur=1,args={'correlation':7}),
        dict(name='_silu_mul_kernel',cat='kernel',ph='X',pid=0,tid=9,ts=30,dur=2,args={'correlation':7})])))
    row,_=analyze(path)
    assert row['attributed_kernels']==1
    assert row['layers'][0]['non_nccl_union_us']==2


def test_streaming_merge_matches_in_memory_timestamps_and_links(tmp_path):
    paths=[]
    for rank in range(2):
        path=tmp_path/f'cuda_rank{rank}.json'
        path.write_text(json.dumps(dict(baseTimeNanoseconds=1000+rank*1000,traceEvents=[
            dict(name='gemm',cat='kernel',ph='X',pid=0,tid=7,ts=20,dur=10,args={'correlation':42})])))
        paths.append(path)
    expected=merge_traces(paths,same_host=True)
    collected=[]
    result=merge_traces(paths,same_host=True,sink=collected.append)
    assert collected==expected['traceEvents']
    assert result['metadata']['summary']==expected['metadata']['summary']
    assert result['traceEvents']==[]


def test_duplicate_rank_files_and_truncated_trace_fail(tmp_path):
    (tmp_path/'cuda_rank0.json').write_text('{"traceEvents":[{"name":')
    with pytest.raises(ValueError,match='truncated'):
        list(TraceReader(tmp_path/'cuda_rank0.json').events())
    (tmp_path/'cuda_rank0.json.gz').touch()
    with pytest.raises(ValueError,match='duplicate'):
        trace_paths(tmp_path)


def test_legacy_layer_ranges_still_label_stage_in_merged_process(tmp_path):
    path=tmp_path/'cuda_rank4.json'
    path.write_text(json.dumps(dict(baseTimeNanoseconds=1,traceEvents=[
        dict(name='stage1.layer24.moe',cat='user_annotation',ph='X',pid=42,tid=42,ts=1,dur=2)])))
    merged=merge_traces([path],same_host=True)
    assert any('rank4 / stage1' in e.get('args',{}).get('name','')
               for e in merged['traceEvents'] if e.get('name')=='process_name')


def capture_fixture(directory):
    """Two messages/two layers, with independent launch IDs and execution list."""
    events = []
    inventory = []
    for index in (1, 2):
        origin = index * 1000
        inventory.append(dict(message_index=index, microbatch_id=index,
                              is_prefill=index == 1, scheduled_sequences=1, succeeded=True))
        phase = 'prefill' if index == 1 else 'decode'
        events.append(dict(name=f'rank0.stage0.message{index}.mb{index}.{phase}',
            cat='user_annotation', ph='X', pid=1, tid=1, ts=origin, dur=500))
        for layer in range(2):
            for op_index, op in enumerate(('attention', 'moe')):
                ts = origin + layer * 100 + op_index * 40 + 10
                corr = index * 100 + layer * 2 + op_index
                for name, cat, start, dur, args in (
                    (f'stage0.layer{layer}.{op}', 'user_annotation', ts, 30, {}),
                    ('aten::mm', 'cpu_op', ts + 1, 5, {'External id': corr}),
                    ('cudaLaunchKernel', 'cuda_runtime', ts + 2, 1, {'correlation': corr}),
                    ('gemm', 'kernel', ts + 40, 10, {'correlation': corr, 'stream': 7})):
                    events.append(dict(name=name, cat=cat, ts=start, dur=dur,
                                       ph='X', pid=1 if cat != 'kernel' else 0, tid=1, args=args))
    path = directory / 'cuda_rank0.json'
    path.write_text(json.dumps(dict(baseTimeNanoseconds=1, traceEvents=events)))
    capture = dict(rank=0, stage=0, layer_range=[0, 2], start_message=1,
                   requested_messages=0, executed_messages=2, captured_messages=inventory,
                   closed_cleanly=True, cuda_activities=True)
    (directory / 'capture_rank0.json').write_text(json.dumps(capture))
    return path, events, capture


@pytest.mark.parametrize('fault', ['none', 'gpu_tail', 'whole_gpu_message', 'whole_message',
                                  'cpu_tail', 'inventory_gap', 'no_manifest', 'duplicate_rank'])
def test_strict_audit_rejects_missing_tail_and_whole_message(tmp_path, fault):
    from utils.trace_audit import validate_traces
    path, events, capture = capture_fixture(tmp_path)
    if fault == 'gpu_tail':
        events = [e for e in events if not (e['cat'] == 'kernel' and e['args']['correlation'] == 203)]
    elif fault == 'whole_gpu_message':
        events = [e for e in events if not (e['cat'] == 'kernel' and e['args']['correlation'] >= 200)]
    elif fault == 'whole_message':
        events = [e for e in events if e['ts'] < 2000]
    elif fault == 'cpu_tail':
        events = [e for e in events if not (e['cat'] == 'user_annotation'
                  and e['name'] == 'stage0.layer1.moe' and e['ts'] > 2000)]
    elif fault == 'inventory_gap':
        capture['captured_messages'].pop()
        (tmp_path / 'capture_rank0.json').write_text(json.dumps(capture))
    elif fault == 'no_manifest':
        (tmp_path / 'capture_rank0.json').unlink()
    elif fault == 'duplicate_rank':
        (tmp_path / 'cuda_rank0.json.gz').touch()
    path.write_text(json.dumps(dict(baseTimeNanoseconds=1, traceEvents=events)))
    result = validate_traces(tmp_path, expected_ranks=1)
    assert result['passed'] == (fault == 'none'), result['errors']
    if fault == 'whole_gpu_message':
        assert result['audits'][0]['message_coverage'][1]['kernel_count'] == 0
        assert any('zero attributed' in e for e in result['errors'])
    assert (tmp_path / 'trace_validation.json').exists()


def test_idle_dp_worker_requires_moe_but_no_attention(tmp_path):
    from utils.trace_audit import validate_traces
    path, events, capture = capture_fixture(tmp_path)
    capture['captured_messages'][1]['scheduled_sequences'] = 0
    events = [e for e in events if not (e['ts'] > 2000 and (
        (e['cat'] == 'user_annotation' and e['name'].endswith('.attention')) or
        (e.get('args', {}).get('correlation', e.get('args', {}).get('External id', -1)) in (200, 202))))]
    path.write_text(json.dumps(dict(traceEvents=events)))
    (tmp_path / 'capture_rank0.json').write_text(json.dumps(capture))
    assert validate_traces(tmp_path, expected_ranks=1)['passed']


def test_nested_legacy_message_range_is_not_counted_twice(tmp_path):
    from utils.trace_audit import validate_traces
    path, events, _ = capture_fixture(tmp_path)
    for index in (1, 2):
        phase = 'prefill' if index == 1 else 'decode'
        events.append(dict(name=f'stage0.microbatch{index}.lane0.{phase}',
            cat='user_annotation', ph='X', pid=1, tid=1, ts=index * 1000 + 1, dur=498))
    path.write_text(json.dumps(dict(baseTimeNanoseconds=1, traceEvents=events)))
    row, _ = analyze(path)
    assert len(row['message_gpu']) == 2
    assert all(m['message'].startswith('rank0.') for m in row['message_gpu'])
    assert validate_traces(tmp_path, expected_ranks=1)['passed']


def test_trace_check_direct_cli_without_pythonpath(tmp_path):
    import os
    import subprocess
    capture_fixture(tmp_path)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'utils/trace_check.py'),
               '--input-dir', str(tmp_path), '--expected-ranks', '1', '--require-cuda']
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['passed']
