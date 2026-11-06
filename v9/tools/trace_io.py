"""Stream Chrome trace events with bounded memory; Python standard library only."""
import gzip
import json
from pathlib import Path
import re


class TraceReader:
    """After events() is exhausted, metadata contains all non-event root fields."""

    def __init__(self, path):
        self.path = Path(path)
        self.metadata = {}

    def events(self):
        opener = gzip.open if self.path.suffix == '.gz' else open
        decoder = json.JSONDecoder()
        with opener(self.path, 'rt', encoding='utf-8-sig') as handle:
            buffer = ''
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    raise ValueError(f'{self.path}: traceEvents array missing')
                buffer += chunk
                match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
                if match:
                    prefix = buffer[:match.end()-1]
                    pos = match.end()
                    break
            while True:
                while pos < len(buffer) and buffer[pos] in ' \r\n\t,':
                    pos += 1
                if pos < len(buffer) and buffer[pos] == ']':
                    self.metadata = json.loads(prefix + '[]' + buffer[pos+1:] + handle.read())
                    self.metadata.pop('traceEvents', None)
                    return
                try:
                    event, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    chunk = handle.read(65536)
                    if not chunk:
                        raise ValueError(f'{self.path}: truncated or invalid event near {buffer[pos:pos+80]!r}')
                    buffer = buffer[pos:] + chunk
                    pos = 0
                    continue
                if not isinstance(event, dict):
                    raise ValueError(f'{self.path}: event is not an object')
                yield event
                pos = end


def trace_paths(directory):
    paths = sorted(Path(directory).glob('cuda_rank*.json*'))
    paths = [p for p in paths if p.name.endswith(('.json', '.json.gz'))]
    ranks = [int(re.search(r'rank(\d+)', p.name)[1]) for p in paths]
    if not paths or len(ranks) != len(set(ranks)):
        raise ValueError('Need one cuda_rankN.json[.gz] per rank; no duplicate JSON/gzip pairs')
    return paths


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(b-a for a, b in union(intervals))


def intersection(a, b):
    a, b = union(a), union(b)
    i = j = 0
    total = 0.
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total
