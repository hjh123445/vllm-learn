"""TP worker：被 run_tp（本机）/ run_tp_crossnode（ssh 到节点 56）调用。
argv: rank tp [master] [port]"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

rank, tp = int(sys.argv[1]), int(sys.argv[2])
# 跨节点 NCCL: 节点 56 的 IB 端口全 DOWN → 显式禁用 IB 走 TCP，
# 并限定 TCP 握手网卡（多网卡机器 NCCL 会选错）
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "bond0")
os.environ["MASTER_ADDR"] = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
os.environ["MASTER_PORT"] = sys.argv[4] if len(sys.argv) > 4 else "29500"
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(tp)
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % 4)))
import torch.distributed as dist
dist.init_process_group("nccl")

from engine.parallel import shard_model_for_tp, set_tp_enabled
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from samplingParams import SamplingParams
from transformers import AutoTokenizer

LOCAL = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
REMOTE = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"
path = REMOTE if os.path.exists(REMOTE + "/config.json") and rank >= 4 else \
    (REMOTE if os.path.exists(REMOTE + "/config.json") and not os.path.exists(LOCAL + "/config.json") else LOCAL)
if not os.path.exists(path + "/config.json"):
    path = REMOTE

tok = AutoTokenizer.from_pretrained(path)
with open(f"{path}/chat_template.json") as f:
    tok.chat_template = json.load(f)["chat_template"]

llm = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                gpu_memory_utilization=0.55, enforce_eager=True,
                tensor_parallel_size=tp)
runner = llm.model_runner
set_tp_enabled(True)
shard_model_for_tp(runner.model, rank, tp)
runner._shard_kv_heads_for_tp(rank, tp)
runner.allocate_kv_cache()
llm.scheduler = Scheduler(llm.config)

msgs = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话"]
ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                              add_generation_prompt=True) for m in msgs]
t0 = time.time()
outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=48))
wall = time.time() - t0
n_tok = sum(len(o["token_ids"]) for o in outs)
if rank == 0:
    print(f"[TP{tp}] wall={wall:.2f}s tokens={n_tok} throughput={n_tok/wall:.1f} tok/s", flush=True)
    print("[TP] 输出:", [o["text"][:36] for o in outs], flush=True)
dist.destroy_process_group()
