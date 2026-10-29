"""PP 启动器: python run_pp.py <pp_size>"""
import os
import subprocess
import sys

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
master = "127.0.0.1"
port = str(29800 + pp)
procs = []
for r in range(pp):
    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(V6, "pp_worker.py"),
         str(r), str(pp), master, port], cwd=V6))
print("exit:", [p.wait() for p in procs])
