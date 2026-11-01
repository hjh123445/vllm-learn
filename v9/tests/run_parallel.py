"""并行配置启动器: python run_parallel.py <tp> <ep>
例: python run_parallel.py 8 1   # TP8 单机
    python run_parallel.py 1 8   # EP8 单机
    python run_parallel.py 4 2   # TP4 × EP2"""
import os
import subprocess
import sys

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tp, ep = int(sys.argv[1]), int(sys.argv[2])
world = tp * ep
master = os.popen("hostname -I").read().split()[0]
port = str(29700 + world * 10 + tp)

procs = []
for r in range(world):
    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(V6, "tests", "parallel_worker.py"),
         str(r), str(world), str(tp), str(ep), master, port], cwd=V6))
rc = [p.wait() for p in procs]
print("exit:", rc)
