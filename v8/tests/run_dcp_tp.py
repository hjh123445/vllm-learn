"""v8.2 混合并行启动器：TP2×DCP2 四卡（ParallelCoords 坐标系统接线引擎）。

验证链（三相位，token 级对比）：
  phase base: 单卡全量（ground truth）
  phase tp:   TP2 双卡（v6.1 配置，DCP off——坐标系统 tp 维）
  phase dcp:  TP2×DCP2 四卡（cp 维块染色 + partial/LSE 合并）
判定：三者的 greedy token 序列逐位一致 + DCP 本地 KV 池减半读数。

用法: python run_dcp_tp.py
"""
import json
import os
import subprocess
import sys

SELF = os.path.abspath(__file__)
V8_DIR = os.path.dirname(os.path.dirname(SELF))
TESTS_DIR = os.path.dirname(SELF)

WORKER = '''
import json, os, sys, time
v8_dir, tests_dir = __V8_DIR__, __TESTS_DIR__
sys.path.insert(0, v8_dir); sys.path.insert(0, tests_dir)
import torch
rank, world, tp, dcp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = sys.argv[5]
os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world)
torch.cuda.set_device(rank)
import torch.distributed as dist
if world > 1:
    # device_id 必须显式绑定——多 new_group 交错的 NCCL bootstrap 竞态会卡死
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{torch.cuda.current_device()}"))

from parallel_coords import init_coords, warmup_groups
coords = init_coords(dp=1, tp=tp, cp=dcp, ep=1, pp=1)
print(f"[r{rank}] {coords}", flush=True)
warmup_groups(coords)

from engine.parallel import shard_model_for_tp, set_tp_enabled, set_tp_group
from layers.dcp import set_dcp
if tp > 1:
    set_tp_enabled(True)
    set_tp_group(coords.group("tp"))
if dcp > 1:
    set_dcp(active=True, size=dcp, rank=coords["cp"], group=coords.group("cp"))

from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from samplingParams import SamplingParams
from transformers import AutoTokenizer

LOCAL = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
REMOTE = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"
path = LOCAL if os.path.exists(LOCAL + "/config.json") else REMOTE
tok = AutoTokenizer.from_pretrained(path)
with open(f"{path}/chat_template.json") as f:
    tok.chat_template = json.load(f)["chat_template"]

util = 0.85 if world == 1 else 0.55
llm = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                gpu_memory_utilization=util, enforce_eager=True,
                tensor_parallel_size=tp)
runner = llm.model_runner
if tp > 1:
    shard_model_for_tp(runner.model, coords["tp"], tp)
    runner._shard_kv_heads_for_tp(coords["tp"], tp)
runner.allocate_kv_cache()
if dcp > 1:
    # DCP: 本地池只留本颜色的块（全局块 g 归 g%dcp==cp_rank 的 rank，
    # 本地槽位 = g//dcp——静态映射保证增量写落在同一物理位置）
    B = llm.config.num_kvcache_blocks
    B_loc = (B + dcp - 1) // dcp
    old = runner.kv_cache
    runner.kv_cache = old[:, :, :B_loc].clone()
    runner._assign_layer_caches(runner.kv_cache)
    del old
    torch.cuda.empty_cache()
    print(f"[r{rank}] DCP 池: 全局 {B} blocks → 本地 {B_loc} "
          f"({runner.kv_cache.numel() * runner.dtype.itemsize / 1e9:.2f} GB)", flush=True)
llm.scheduler = Scheduler(llm.config)   # BlockManager 用全局块数视图

msgs = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话"]
ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                              add_generation_prompt=True) for m in msgs]
t0 = time.time()
outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=48))
wall = time.time() - t0
n_tok = sum(len(o["token_ids"]) for o in outs)
if rank == 0:
    with open(sys.argv[6], "w") as f:
        json.dump({"tokens": [o["token_ids"] for o in outs],
                   "texts": [o["text"] for o in outs]}, f, ensure_ascii=False)
    print(f"[{sys.argv[7]}] wall={wall:.2f}s tok={n_tok} thr={n_tok/wall:.1f} tok/s", flush=True)
    print("[OUT]", [o["text"][:36] for o in outs], flush=True)
if world > 1:
    dist.destroy_process_group()
'''


def run_phase(name, world, tp, dcp, out, port, gpus):
    src = (WORKER.replace("__V8_DIR__", repr(V8_DIR))
                .replace("__TESTS_DIR__", repr(TESTS_DIR)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus))
    procs = []
    for r in range(world):
        procs.append(subprocess.Popen(
            [sys.executable, "-c", src, str(r), str(world), str(tp), str(dcp),
             str(port), out, name], env=env, cwd=V8_DIR))
    rcs = [p.wait() for p in procs]
    assert all(rc == 0 for rc in rcs), f"{name} phase failed: {rcs}"
    return json.load(open(out))


def main():
    base = run_phase("base", 1, 1, 1, "/tmp/v82_base.json", 29931, ["2"])
    tp = run_phase("tp", 2, 2, 1, "/tmp/v82_tp.json", 29932, ["2", "3"])
    dcp = run_phase("dcp", 4, 2, 2, "/tmp/v82_dcp.json", 29933, ["2", "3", "4", "5"])

    print("\n===== v8.2 正确性裁决（greedy token 逐位对比）=====")
    for name, r in (("TP2", tp), ("TP2×DCP2", dcp)):
        same = all(a == b for a, b in zip(base["tokens"], r["tokens"]))
        n_eq = sum(sum(x == y for x, y in zip(a, b)) for a, b in zip(base["tokens"], r["tokens"]))
        n_tot = sum(len(a) for a in base["tokens"])
        print(f"  base vs {name}: {'ALL MATCH' if same else 'MISMATCH'} "
              f"({n_eq}/{n_tot} tokens)")
        if not same:
            for i, (a, b) in enumerate(zip(base["tokens"], r["tokens"])):
                if a != b:
                    j = next(k for k in range(min(len(a), len(b))) if a[k] != b[k])
                    print(f"    req{i} 首个分歧 @tok{j}: {a[j]} vs {b[j]}")
    print("\n===== v8.2 完成 =====")


if __name__ == "__main__":
    main()
