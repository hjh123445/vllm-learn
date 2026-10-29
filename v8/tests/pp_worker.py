"""v6 PP worker：流水线并行（层切分）。

架构（rank = PP stage）：
  - stage 0 持 embedding + 前 L/N 层 + 首层前的处理
  - stage k 持第 k 段层
  - 最后 stage 持末层 + final norm + lm_head + 采样
  - 层边界 p2p: stage i 把 hidden [T, H] 发给 stage i+1（torch P2POp 批量流水）
  - micro-batch: 请求切成 M 份，stage i 算 mb_j 时 stage i+1 算 mb_{j-1}（重叠）

KV cache: 各 stage 只存自己那段的层 ✓（显存按层数均摊）
调度: 所有 stage 跑同样的 scheduler（输入相同决策相同——与 TP/EP 同假设）
用法: python run_pp.py <pp_size>   （单机多进程）
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

rank, pp = int(sys.argv[1]), int(sys.argv[2])
os.environ["MASTER_ADDR"] = sys.argv[3]
os.environ["MASTER_PORT"] = sys.argv[4]
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(pp)
torch.cuda.set_device(rank)
import torch.distributed as dist
import torch.distributed.pipelining as pip

dist.init_process_group("nccl")
os.environ["BROADCAST_SRC"] = str(pp - 1)   # 真 token 在最后 stage

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

# 完整引擎构造（每 stage 都全量构造+加载——PP 切层在后面，显存只留自己的段）
# tensor_parallel_size=pp 复用"跳过 warmup+跳过分配"机制（切层后手动分池）
llm = LLMEngine(path, max_num_seqs=8, max_model_len=2048,
                gpu_memory_utilization=0.92, enforce_eager=True,
                tensor_parallel_size=pp)
runner = llm.model_runner
model = runner.model
hf = runner.config.hf_config

# ---- PP 切层：只保留自己的段，其余层替换为 Identity（显存释放）----
L = hf.num_hidden_layers
per = L // pp
# 只保留自己的段（旧引用随 ModuleList 替换释放）
keep = [model.model.layers[i] for i in range(rank * per, (rank + 1) * per)]
model.model.layers = torch.nn.ModuleList(keep)
torch.cuda.empty_cache()

# embedding 只在 stage 0 / lm_head+norm 只在最后 stage
if rank != 0:
    model.model.embed_tokens = None
if rank != pp - 1:
    model.lm_head = None
    model.model.norm = None

# ---- KV cache: 池子层数 = per（只存我的段）----
runner.config.hf_config.num_hidden_layers = per         # 池分配按此
runner.allocate_kv_cache()
# 池子大小跨 stage 对齐（不均 → can_allocate 分叉 → p2p 错位死锁）:
# 所有 stage 用同一池子大小（取 min）
import torch.distributed as dist
nblk = torch.tensor([runner.config.num_kvcache_blocks], device="cuda")
dist.all_reduce(nblk, op=dist.ReduceOp.MIN)
if int(nblk.item()) != runner.config.num_kvcache_blocks:
    # 重建更小的池子
    runner.config.num_kvcache_blocks = int(nblk.item())
    from engine.model_runner import ModelRunner
    hf2 = runner.config.hf_config
    num_kv_heads = getattr(runner, "_tp_kv_heads", None) or hf2.num_key_value_heads
    head_dim = getattr(hf2, "head_dim", hf2.hidden_size // hf2.num_attention_heads)
    runner.kv_cache = torch.empty(2, per, runner.config.num_kvcache_blocks,
                                  runner.block_size, num_kv_heads, head_dim,
                                  dtype=runner.dtype, device="cuda")
    runner._assign_layer_caches(runner.kv_cache)
# scheduler 池对齐（池对齐后重建——Scheduler 构造时从 config 读块数）
llm.scheduler = Scheduler(llm.config)

# ---- 前向改为 stage 化 ----
from utils.context import get_context, set_context

_pending_sends = []


def stage_forward(input_ids, positions):
    """本 stage 的前向：recv 上游 hidden（或 embedding）→ 我的层 → send 下游。"""
    global _pending_sends
    # 不 wait isend——它的完成依赖对端 recv，而 rank1 可能已推进到 broadcast
    # （死锁闭环: wait←recv←broadcast←wait）。contiguous 拷贝保证缓冲独立，
    # 句柄只用于退出前 drain
    _pending_sends = []
    ctx = get_context()
    if rank == 0:
        x = model.model.embed_tokens(input_ids)
    else:
        x = torch.empty(input_ids.shape[0], hf.hidden_size, device="cuda",
                        dtype=runner.dtype)
        dist.recv(x, src=rank - 1)   # p2p: 阻塞直到对端 send
    residual = torch.zeros_like(x)
    for layer in model.model.layers:
        x, residual = layer(x, residual, positions)
    if rank < pp - 1:
        # 发 hidden 和 residual（下游要接续残差流）。
        # isend + 立即返回（send 会等对端 recv 配对 → 与下游的 broadcast 死锁），
        # 请求句柄存全局，下一步 forward 前统一 wait
        _pending_sends += [dist.isend(x.contiguous(), dst=rank + 1),
                           dist.isend(residual.contiguous(), dst=rank + 1)]
        return None
    # 最后 stage: 最后一层输出已含残差，norm 直接作用其上（与单卡收尾一致）
    return model.model.norm(x)


# 覆盖 model_runner.run 的模型调用部分：直接换 run_model
orig_run = runner.run

def pp_run(seqs, is_prefill):
    from engine.sequence import Sequence
    # 所有 stage 构造同样的输入（schedule 相同）
    if is_prefill:
        input_ids, positions = runner.prepare_prefill(seqs)
    else:
        input_ids, positions = runner.prepare_decode(seqs)
    hidden = stage_forward(input_ids, positions)
    if rank == pp - 1:
        # prefill 取每序列最后位置 / decode 全部
        if is_prefill:
            ctx = get_context()
            last = ctx.cu_seqlens_q[1:] - 1
            logits = model.lm_head(hidden[last])
        else:
            logits = model.lm_head(hidden)
        temperatures = torch.tensor([s.temperature for s in seqs],
                                    dtype=torch.float32, device="cuda")
        return runner.sampler(logits, temperatures).tolist()
    return [0] * len(seqs)   # 非最后 stage 返回占位（broadcast 会覆盖）

runner.run = pp_run

msgs = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话"]
ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                              add_generation_prompt=True) for m in msgs]
t0 = time.time()
outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=48))
wall = time.time() - t0
n_tok = sum(len(o["token_ids"]) for o in outs)
if rank == pp - 1:
    print(f"[PP{pp}] wall={wall:.2f}s tok={n_tok} thr={n_tok/wall:.1f} tok/s", flush=True)
    print("[OUT]", [o["text"][:30] for o in outs], flush=True)
dist.destroy_process_group()
