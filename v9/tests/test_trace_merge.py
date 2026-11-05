"""Offline contracts for distributed timeline alignment; no GPU required."""
import json
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("trace_merge", Path(__file__).with_name("merge_pipeline_trace.py"))
merger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merger)


def save(tmp_path, rank, events, **metadata):
    path = tmp_path / f"cuda_rank{rank}.json"
    path.write_text(json.dumps({"traceEvents": events, **metadata}), encoding="utf-8")
    return path


def kernel(ts=100, dur=10, tid=7, name="gemm", **kwargs):
    return {"ph": "X", "pid": 0, "tid": tid, "cat": "kernel", "name": name,
            "ts": ts, "dur": dur, "args": kwargs}


def test_overlay_preserves_cross_rank_concurrency_and_unique_tracks(tmp_path):
    paths = [save(tmp_path, rank, [kernel()], baseTimeNanoseconds=1000000) for rank in (0, 1)]
    trace = merger.merge_traces(paths, same_host=True)
    events = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    assert [e["ts"] for e in events] == [0, 0]
    assert [e["dur"] for e in events] == [10, 10]
    assert events[0]["pid"] != events[1]["pid"]
    assert events[0]["tid"] != events[1]["tid"]


def test_reconciles_different_bases_before_one_common_origin(tmp_path):
    a = save(tmp_path, 0, [kernel(ts=100)], baseTimeNanoseconds=1000000)
    b = save(tmp_path, 1, [kernel(ts=95)], baseTimeNanoseconds=1010000)
    events = merger.merge_traces([a, b], same_host=True)["traceEvents"]
    assert [e["ts"] for e in events if e.get("cat") == "kernel"] == [0, 5]


def test_ids_keep_cpu_gpu_and_flow_links_within_source_only(tmp_path):
    raw = [kernel(**{"External id": 42, "correlation": 123}),
           {"ph": "X", "pid": 999, "tid": 999, "cat": "cpu_op", "name": "aten::mm",
            "ts": 99, "dur": 2, "args": {"External id": 42, "correlation": 123}},
           {"ph": "s", "pid": 999, "tid": 999, "ts": 100, "id": 10, "cat": "ac2g", "name": "flow"},
           {"ph": "f", "pid": 0, "tid": 7, "ts": 100, "id": 10, "cat": "ac2g", "name": "flow"}]
    paths = [save(tmp_path, rank, raw, baseTimeNanoseconds=1000000) for rank in (0, 1)]
    events = merger.merge_traces(paths, same_host=True)["traceEvents"]
    links = []
    for path in paths:
        source = [e for e in events if e.get("args", {}).get("trace_source") == path.name]
        assert source[0]["args"]["External id"] == source[1]["args"]["External id"]
        assert source[0]["args"]["correlation"] == source[1]["args"]["correlation"]
        assert source[2]["id"] == source[3]["id"]
        links.append((source[0]["args"]["External id"], source[2]["id"]))
    assert links[0][0] != links[1][0] and links[0][1] != links[1][1]


def test_gpu_overlap_uses_union_and_ignores_nested_cpu_scopes(tmp_path):
    raw = [kernel(100, 10), kernel(102, 6, tid=8), kernel(105, 10, tid=9, name="ncclAllToAll"),
           {"ph": "X", "pid": 999, "tid": 999, "cat": "forward", "name": "host",
            "ts": 90, "dur": 100000}]
    row = merger.merge_traces([save(tmp_path, 0, raw)])["metadata"]["summary"]["gpu_tracks"][0]
    assert row["non_nccl_kernel_union_us"] == 10
    assert row["nccl_kernel_union_us"] == 10
    assert row["gpu_active_union_us"] == 15
    assert row["nccl_non_nccl_intersection_us"] == 5
    assert row["fraction_nccl_time_overlapped"] == .5


def test_mixed_clock_domains_rejected(tmp_path):
    a = save(tmp_path, 0, [kernel()], baseTimeNanoseconds=1000000)
    b = save(tmp_path, 1, [], metadata={"clock": "host_monotonic_us"})
    with pytest.raises(ValueError, match="Mixed clock"):
        merger.merge_traces([a, b], same_host=True)


def test_cross_host_and_unspecified_clocks_require_explicit_evidence(tmp_path):
    a, b = [save(tmp_path, rank, [kernel()]) for rank in (0, 1)]
    with pytest.raises(ValueError, match="same-host"):
        merger.merge_traces([a, b])
    with pytest.raises(ValueError, match="Missing time bases"):
        merger.merge_traces([a, b], same_host=True)
    assert merger.merge_traces([a, b], same_host=True, shared_timestamps=True)


def test_no_gpu_activity_is_not_reported_as_compute(tmp_path):
    raw = [{"ph": "X", "pid": 0, "tid": "compute", "cat": "forward", "name": "forward",
            "ts": 100, "dur": 500}]
    trace = merger.merge_traces([save(tmp_path, 0, raw, metadata={"clock": "host_monotonic_us"})])
    assert trace["metadata"]["summary"]["gpu_tracks"] == []
    assert trace["metadata"]["workers"][0]["gpu_event_count"] == 0


def test_independent_dp_trace_names_override_world_local_rank(tmp_path):
    path = save(tmp_path, 4, [kernel()], distributedInfo={"rank": 0})
    assert merger.merge_traces([path])["metadata"]["workers"][0]["rank"] == 4


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="No input"):
        merger.merge_traces([])
