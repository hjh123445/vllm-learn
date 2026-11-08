"""Persist the command, child PID and terminal result for a remote experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--state', required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
path = Path(args.state)
path.parent.mkdir(parents=True, exist_ok=True)
state = dict(command=command, started=time.time(), status='starting')
def write():
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)
write()
try:
    child = subprocess.Popen(command)
    state.update(pid=child.pid, status='running')
    write()
    code = child.wait()
    state.update(status='passed' if code == 0 else 'failed', returncode=code, ended=time.time())
except Exception as error:
    state.update(status='failed', error=repr(error), ended=time.time())
    code = 1
write()
raise SystemExit(code)
