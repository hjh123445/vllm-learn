"""v6.4 worker：集中化调度的 EP 验证（EP4/EP8——之前的 open issue）。

与 tests/parallel_worker.py 的区别：step 被 centralized.make_centralized_step
替换——rank0 独占 scheduler，follower 只跟随计划。用法:
    python test_centralized.py <rank> <world> <tp> <ep> <master> <port>
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

rank, world, tp, ep = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
os.environ["MASTER_ADDR"] = sys.argv[5]
os.environ["MASTER_PORT"] = sys.argv[6]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world)
torch.cuda.set_device(rank)
import torch.distributed as dist
dist.init_process_group("nccl")

from engine.parallel import shard_model_for_tp, shard_model_for_ep, set_tp_group, set_tp_enabled
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from engine.centralized import make_centralized_step
from samplingParams import SamplingParams
from transformers import AutoTokenizer

tp_rank, ep_rank = rank // ep, rank % ep
if ep > 1 and tp > 1:
    tp_group = dist.new_group(ranks=[r for r in range(world) if r // ep == rank // ep])
    set_tp_group(tp_group)

path = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"

tok = AutoTokenizer.from_pretrained(path)
with open(f"{path}/chat_template.json") as f:
    tok.chat_template = json.load(f)["chat_template"]

llm = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                gpu_memory_utilization=0.8, enforce_eager=True,
                tensor_parallel_size=tp if tp > 1 else 1,
                expert_parallel_size=ep if ep > 1 else 1)
runner = llm.model_runner
if tp > 1:
    set_tp_enabled(True)
    shard_model_for_tp(runner.model, tp_rank, tp)
    runner._shard_kv_heads_for_tp(tp_rank, tp)
if ep > 1:
    shard_model_for_ep(runner.model, ep_rank, ep)
runner.allocate_kv_cache()
llm.scheduler = Scheduler(llm.config)

# ===== v6.4 核心：替换 step 为集中化哑 forward 版本 =====
import os as _os
if _os.environ.get("CONTROL"):
    class _FakeCoord:
        is_master = True
        _seqs = {}
        def sync_sequences(self, seqs): self._seqs = {s.seq_id: s for s in seqs}
    coord = _FakeCoord()
else:
    llm.step, coord = make_centralized_step(llm)

msgs = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话"]
ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                              add_generation_prompt=True) for m in msgs]

# 手动驱动（走我们替换过的 step）——保留 seq 引用供收集
my_seqs = [llm.add_request(p, SamplingParams(temperature=0.0, max_tokens=48)) for p in ps]

t0 = time.time()
while not llm.is_finished():
    llm.step()
wall = time.time() - t0

# master 发终止计划（空批）——follower 收到 n=0 才能退出循环
if _os.environ.get("CONTROL"):
    pass
else:
    # 先等最后一步 forward 的异步 allreduce 落地，再发终止通知——
    # 否则两者抢 communicator 会乱序（收尾死锁的根因）
    torch.cuda.synchronize()
    coord.plan_buf[:, 0] = -1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(coord.plan_buf, src=coord.master)
        torch.distributed.broadcast(coord.bt_buf, src=coord.master)
        torch.distributed.broadcast(coord.tok_buf, src=coord.master)
        torch.cuda.synchronize()

if rank == 0:
    print(f"[TP{tp}xEP{ep}-cent] wall={wall:.2f}s", flush=True)

# 从各自视角打印（follower 的 completion 也镜像了）
for rr in (0, world - 1):
    if rank == rr:
        print(f"=== r{rank} ===", flush=True)
        for j, s in enumerate(my_seqs):
            txt = tok.decode(s.completion_token_ids, skip_special_tokens=True)
            print(f"  req{j} ({len(s.completion_token_ids)} tok): {txt[:36]!r}", flush=True)
    dist.barrier()

dist.barrier()
dist.destroy_process_group()
