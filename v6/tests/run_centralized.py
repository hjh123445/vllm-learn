"""集中化 EP 启动器: python run_centralized.py <ep>"""
import os
import subprocess
import sys

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 4
master = "127.0.0.1"
port = str(29850 + ep)
procs = []
for r in range(ep):
    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(V6, "tests", "test_centralized.py"),
         str(r), str(ep), "1", str(ep), master, port], cwd=V6))
print("exit:", [p.wait() for p in procs])
