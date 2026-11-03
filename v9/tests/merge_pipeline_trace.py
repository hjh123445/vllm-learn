"""Merge per-worker PP traces and compute overlap statistics.

Usage:
  python tests/merge_pipeline_trace.py --input-dir trace_baseline --output trace.json
"""
import argparse
import json
import os
from collections import defaultdict


def load_events(input_dir):
    rows = []
    metadata = []
    for name in sorted(os.listdir(input_dir)):
        if not (name.startswith("pipeline_rank") or name.startswith("cuda_rank")) \
                or not name.endswith(".json"):
            continue
        with open(os.path.join(input_dir, name), encoding="utf-8") as f:
            data = json.load(f)
        rows.extend(data.get("traceEvents", []))
        metadata.append({"file": name, **data.get("metadata", {})})
    return rows, metadata


def summarize(events):
    compute = defaultdict(float)
    comm = defaultdict(float)
    for event in events:
        duration = float(event.get("dur", 0.0))
        stage = event.get("args", {}).get("stage", -1)
        if event.get("cat") == "forward":
            compute[stage] += duration
        elif event.get("cat") == "p2p":
            comm[stage] += duration
    return {
        "compute_ms_by_stage": {str(k): v / 1000 for k, v in compute.items()},
        "communication_ms_by_stage": {str(k): v / 1000 for k, v in comm.items()},
        "event_count": len(events),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    events, metadata = load_events(args.input_dir)
    result = {
        "displayTimeUnit": "ms",
        "traceEvents": events,
        "metadata": {
            "source_dir": os.path.abspath(args.input_dir),
            "workers": metadata,
            "summary": summarize(events),
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(json.dumps(result["metadata"]["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
