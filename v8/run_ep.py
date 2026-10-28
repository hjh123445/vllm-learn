"""v6.2 EP 启动器（文件版 worker，跨节点可用）。
用法: python run_ep.py [cross]"""
import os
import subprocess
import sys

V6 = os.path.dirname(os.path.abspath(__file__))
NODE1 = "172.28.30.56"


def main():
    cross = len(sys.argv) > 1 and sys.argv[1] == "cross"
    ep = 2
    master = os.popen("hostname -I").read().split()[0]
    port = "29610"
    env0 = dict(os.environ, LOCAL_RANK="0")
    p0 = subprocess.Popen([sys.executable, os.path.join(V6, "ep_worker.py"),
                           "0", str(ep), master, port], cwd=V6, env=env0)
    if cross:
        PY = "/home/ifly_jyhe23/env/miniconda3/bin/python"
        remote = (f"cd {V6} && LOCAL_RANK=0 nohup {PY} {V6}/ep_worker.py 1 {ep} "
                  f"{master} {port} > /tmp/ep_r1.log 2>&1 &")
        subprocess.run(["ssh", NODE1, remote], check=False)
    else:
        env1 = dict(os.environ, LOCAL_RANK="1")
        subprocess.Popen([sys.executable, os.path.join(V6, "ep_worker.py"),
                          "1", str(ep), master, port], cwd=V6, env=env1)
    p0.wait()


if __name__ == "__main__":
    main()
