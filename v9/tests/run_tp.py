"""v6.1 TP 启动器（独立进程版）。

⚠ 踩坑记录（ENGINEERING_NOTES）: 早期版本用 mp.Process spawn + mp.Manager
传参——同一套代码在 spawn 子进程里稳定输出乱码（prefill 采样就错），
换成两个独立 python 进程后完全正确。根因未深究（疑似 Manager pickle
与子进程 CUDA 初始化的时序交互），保守方案: 每卡一个独立进程,
由启动脚本 fork，结果通过文件交换。

用法: python run_tp.py [tp_size]  → 内部 self-exec 两个独立进程
"""
import os
import subprocess
import sys

SELF = os.path.abspath(__file__)

WORKER = '''
import os, sys
sys.path.insert(0, __V6_DIR__)
import torch
rank = int(sys.argv[1]); tp = int(sys.argv[2])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = os.environ.get("TP_PORT", "29500")
os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(tp)
torch.cuda.set_device(rank)
import torch.distributed as dist
dist.init_process_group("nccl")
from engine.parallel import init_distributed, shard_model_for_tp
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from samplingParams import SamplingParams
import json
from transformers import AutoTokenizer

path = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
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
outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=32))
if rank == 0:
    with open("/tmp/tp_out.json", "w") as f:
        json.dump([o["text"][:60] for o in outs], f, ensure_ascii=False)
    print("[TP] 输出:", [o["text"][:40] for o in outs], flush=True)
dist.destroy_process_group()
'''


def main():
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    v6_dir = os.path.dirname(SELF)
    worker_src = WORKER.replace("__V6_DIR__", repr(v6_dir))
    procs = []
    for r in range(tp):
        p = subprocess.Popen([sys.executable, "-c", worker_src, str(r), str(tp)],
                             cwd=v6_dir)
        procs.append(p)
    rc = [p.wait() for p in procs]
    print("exit codes:", rc)


if __name__ == "__main__":
    main()
