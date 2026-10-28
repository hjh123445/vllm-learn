"""v6.2 跨节点 TP 启动器：本机 N 卡 + 节点 56 N 卡。

验证 PARALLEL_ANALYSIS 的核心论断: TP 跨节点每层 allreduce 走 IB（40Gb/s）,
TPOT 会显著劣化于节点内 TP——这个脚本同时量两种配置的 TPOT 做对比。

用法: python run_tp_crossnode.py          # 本机4 + 节点56 4 = TP8
      python run_tp_crossnode.py local    # 仅本机 TP4 对照组
"""
import os
import subprocess
import sys

NODE1 = "172.28.30.56"
PORT = "29600"
V6 = os.path.dirname(os.path.abspath(__file__))

WORKER = '''
import os, sys, time
sys.path.insert(0, __V6_DIR__)
import torch
rank = int(sys.argv[1]); tp = int(sys.argv[2])
os.environ["MASTER_ADDR"] = "__MASTER__"
os.environ["MASTER_PORT"] = "__PORT__"
os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(tp)
local_rank = rank % (tp // 2) if __CROSS__ else rank
torch.cuda.set_device(local_rank)
import torch.distributed as dist
dist.init_process_group("nccl")
from engine.parallel import shard_model_for_tp
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from samplingParams import SamplingParams
import json
from transformers import AutoTokenizer

path = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct" if rank >= 4 else "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
tok = AutoTokenizer.from_pretrained(path)
with open(f"{path}/chat_template.json") as f:
    tok.chat_template = json.load(f)["chat_template"]

llm = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                gpu_memory_utilization=0.55, enforce_eager=True,
                tensor_parallel_size=tp)
runner = llm.model_runner
shard_model_for_tp(runner.model, rank, tp)
runner._shard_kv_heads_for_tp(rank, tp)
runner.allocate_kv_cache()
llm.scheduler = Scheduler(llm.config)

msgs = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话"]
ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                              add_generation_prompt=True) for m in msgs]
sp = SamplingParams(temperature=0.0, max_tokens=48)
t0 = time.time()
outs = llm.generate(ps, sp)
wall = time.time() - t0
n_tok = sum(len(o["token_ids"]) for o in outs)
if rank == 0:
    print(f"[TP{tp}] wall={wall:.2f}s tokens={n_tok} throughput={n_tok/wall:.1f} tok/s", flush=True)
    print("[TP] 输出:", [o["text"][:36] for o in outs], flush=True)
dist.destroy_process_group()
'''


def main():
    cross = len(sys.argv) < 2 or sys.argv[1] != "local"
    tp_per_node = 4
    tp = tp_per_node * 2 if cross else tp_per_node
    master = os.popen("hostname -I").read().split()[0]
    src = (WORKER.replace("__V6_DIR__", repr(V6))
                .replace("__MASTER__", master)
                .replace("__PORT__", PORT)
                .replace("__CROSS__", "True" if cross else "False"))
    procs = []
    for r in range(tp_per_node):           # 本机
        env = dict(os.environ, LOCAL_RANK=str(r), NCCL_DEBUG="WARN")
        procs.append(subprocess.Popen([sys.executable, os.path.join(V6, "tp_worker.py"),
                                       str(r), str(tp), master, PORT], cwd=V6, env=env))
    if cross:
        # 节点 56: 代码已 rsync 同步（共享目录路径一致），直接 ssh 起对端 worker
        PY = "/home/ifly_jyhe23/env/miniconda3/bin/python"
        worker_file = os.path.join(V6, "tp_worker.py")
        remote = (f"cd {V6} && for r in {tp_per_node} {tp_per_node+1} {tp_per_node+2} {tp_per_node+3}; do "
                  f"nohup {PY} {worker_file} $r {tp} {master} {PORT} > /tmp/tp_r$r.log 2>&1 & done")
        subprocess.run(["ssh", NODE1, remote], check=False)
    rc = [p.wait() for p in procs]
    print("local exit codes:", rc)


if __name__ == "__main__":
    main()
