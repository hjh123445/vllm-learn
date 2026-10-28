"""并行 worker: rank world tp ep master port
TP 和 EP 可同时开（TP 切层内矩阵, EP 切专家维——正交）。
例如 8 卡 = TP2×EP4（2 卡一组张量并行 × 4 组各持 1/4 专家）。"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
rank, world, tp, ep = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
assert tp * ep == world, f"tp({tp}) x ep({ep}) != world({world})"
os.environ["MASTER_ADDR"] = sys.argv[5]
os.environ["MASTER_PORT"] = sys.argv[6]
os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world)
torch.cuda.set_device(rank)
import torch.distributed as dist
dist.init_process_group("nccl")

# TP 组: rank // ep 同组（连续 ep 个 rank 是一组 TP）
tp_group = dist.new_group(ranks=[r for r in range(world) if r // ep == rank // ep])
# EP 组: rank % ep... 不对——EP 的 expert 切分是跨 TP 组的同一个"层副本"位置
# 简化拓扑: rank = tp_rank * ep + ep_rank（tp_rank 选 TP 组, ep_rank 选专家片）
tp_rank, ep_rank = rank // ep, rank % ep

from engine.parallel import shard_model_for_tp, set_tp_enabled, shard_model_for_ep, set_tp_group
# TP 组覆盖全部 rank 时不需要子组（world group 即 TP 组）
if ep > 1:
    set_tp_group(tp_group)
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

llm = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                gpu_memory_utilization=0.8, enforce_eager=True,
                tensor_parallel_size=tp if tp > 1 else 1,
                expert_parallel_size=ep if ep > 1 else 1)
runner = llm.model_runner
if tp > 1:
    set_tp_enabled(True)
set_tp_enabled(True)
shard_model_for_tp(runner.model, tp_rank, tp)
    runner._shard_kv_heads_for_tp(tp_rank, tp)
if ep > 1:
    shard_model_for_ep(runner.model, ep_rank, ep)
runner.allocate_kv_cache()
llm.scheduler = Scheduler(llm.config)

# 单一 rank0 跑 scheduler 决策（TP/EP 下所有 rank 必须以相同顺序前向——
# 简化: 所有 rank 跑同样的 generate（输入相同 → 决策相同, 已验证））
msgs = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话"]
ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                              add_generation_prompt=True) for m in msgs]
t0 = time.time()
outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=48))
wall = time.time() - t0
n_tok = sum(len(o["token_ids"]) for o in outs)
if rank == 0:
    print(f"[TP{tp}xEP{ep}] wall={wall:.2f}s tok={n_tok} thr={n_tok/wall:.1f} tok/s", flush=True)
    print("[OUT]", [o["text"][:30] for o in outs], flush=True)
dist.destroy_process_group()
