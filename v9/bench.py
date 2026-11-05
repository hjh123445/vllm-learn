"""Unified CLI: offline engine matrix or online OpenAI-compatible HTTP load.

python v9/bench.py offline --help
python v9/bench.py serve --help
"""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('offline', 'serve', 'trace-check'))
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        parser.print_help()
        return
    mode = sys.argv[1]
    if mode not in ('offline', 'serve', 'trace-check'):
        parser.error('mode must be offline, serve or trace-check')
    # Lazy import: help/HTTP client does not initialize torch or load a model.
    if mode == 'offline':
        path = Path(__file__).parent / 'tests' / 'bench_workload_matrix.py'
    elif mode == 'serve':
        from utils.http_benchmark import main as run
    else:
        from utils.trace_check import main as run
    sys.argv = [sys.argv[0] + ' ' + mode, *sys.argv[2:]]
    if mode == 'offline':
        runpy.run_path(str(path), run_name='__main__')
    else:
        run()


if __name__ == '__main__':
    main()
