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
