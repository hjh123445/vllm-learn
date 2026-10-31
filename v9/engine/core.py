# v9/engine/core.py
"""v9.4 EngineCore：scheduler + executor 同进程（vLLM v1 形态的核心）。

对标 vLLM: v1/engine/core.py 的 EngineCore（scheduler 唯一决策者）+
v1/executor/multiproc_executor.py 的 MultiprocExecutor（spawn WorkerProc）。

busy loop（对齐 vLLM 的 step 语义）:
    scheduler.schedule() → SchedulerOutput 广播给所有 worker
    → 等所有 worker 的 response（TP/EP 下 forward 内部已对齐，
      response 只从 rank0 取——它已聚合完整 logits 的采样结果）
    → scheduler.update_from_output（完成判定/状态推进）

单卡退化（对齐 UniProcExecutor）: world_size==1 时不 spawn 子进程，
ModelRunner 同进程直调——同一条 EngineCore 代码路径。

与 v6.4 centralized 的对照（VLLM_ARCH.md）: scheduler 从"rank0 进程兼任"
变成"EngineCore 主进程独占"，worker 从"镜像状态的完整引擎"变成"无状态
forward 执行器"，通信从固定形状 NCCL 张量变成结构化 MQ 消息。
"""

import multiprocessing as mp
import os

import torch


def _spawn_with_env(env, fn, args):
    """spawn 的 env 传递：mp.Process 不支持 per-child env，包一层。
    ⚠ 只覆盖目标键不 clear——spawn 的 bootstrap 依赖 mp 自己注入的环境变量，
    clear 会让子进程在 import 前死掉（zombie、无 stderr——v9.4a 实测）。"""
    os.environ.update(env)
    fn(*args)

from engine.mq import SHUTDOWN, make_channels
from engine.sequence import Sequence
from samplingParams import SamplingParams


def _spec_msg(seqs, is_prefill):
    """scheduler 的输出 → 可 pickle 的结构化消息。

    对齐 vLLM 的 SchedulerOutput：scheduled（要 forward 的序列描述）+
    finished（本步完成）。SeqSpec 是【用完即弃的快照】——worker 不维护
    跨步状态（v6.4 _SeqSpec 的职责平移，但生命周期只到这次 forward）。"""
    scheduled = []
    for s in seqs:
        start = s.num_cached_tokens
        end = start + s.num_scheduled_tokens
        scheduled.append({
            "seq_id": s.seq_id,
            "is_prefill": s.is_prefill,
            "num_cached_tokens": s.num_cached_tokens,
            "num_scheduled_tokens": s.num_scheduled_tokens,
            "block_table": list(s.block_table),
            "tokens": s[start:end],
            "last_token": s.last_token,
            "seq_len": len(s),
            "temperature": s.temperature,
        })
    return {"type": "schedule", "scheduled": scheduled, "is_prefill": is_prefill}


class _Spec:
    """worker 侧从消息重建的 forward 输入（model_runner.run 的参数形态）。

    model_runner 期望 Sequence 的鸭子类型子集——直接补齐它读的字段，
    不继承 Sequence（不携带任何跨步状态）。"""

    def __init__(self, m):
        self.seq_id = m["seq_id"]
        self.is_prefill = m["is_prefill"]
        self.num_cached_tokens = m["num_cached_tokens"]
        self.num_scheduled_tokens = m["num_scheduled_tokens"]
        self.block_table = m["block_table"]
        self._tokens = m["tokens"]
        self.last_token = m["last_token"]
        self.temperature = m["temperature"]
        bs = 256
        n = m["seq_len"]
        self.num_tokens = n
        self.num_prompt_tokens = n          # 哑视角够用（run 只读下列字段）
        self.last_block_num_tokens = n - (len(self.block_table) - 1) * bs \
            if self.block_table else n

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, k):
        return self._tokens[k]


def rebuild_specs(msg):
    return [_Spec(m) for m in msg["scheduled"]]


class EngineCore:
    """v9.4 的引擎核心。对外暴露与 LLMEngine.generate 兼容的批推理入口。"""

    def __init__(self, model_path, tp_size=1, ep_size=1, dcp_size=1, pp_size=1,
                 master_port=29600, config_kwargs=None, gpus=None):
        self.world_size = tp_size * ep_size * dcp_size * pp_size
        self.tp_size, self.ep_size = tp_size, ep_size

        # ---- 顺序关键：runner 先于 scheduler ----
        from engine.scheduler import Scheduler
        # BlockManager 的块数来自 config.num_kvcache_blocks，那是 ModelRunner
        # warmup+分配之后才填的（v8 的 llm_engine 就是这个顺序；v9.4a 第一版
        # scheduler 在前 → BlockManager(−1→0 块) → schedule 直接 assert）
        from config import Config
        config_kwargs = dict(config_kwargs or {})
        config = Config(model_path, **config_kwargs)
        self.config = config
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        eos = self.tokenizer.eos_token_id
        config.eos = eos[0] if isinstance(eos, list) else eos

        engine_kwargs = {
            "master_addr": "127.0.0.1", "master_port": master_port,
            "config_kwargs": config_kwargs, "dcp_size": dcp_size, "gpus": gpus,
            "pp_size": pp_size,
        }
        self.pp_size = pp_size
        self.dcp_size = dcp_size

        if self.world_size == 1:
            # UniProc 路径：ModelRunner 同进程直调（分配池子 → 填 config）
            from engine.model_runner import ModelRunner
            self.inproc_runner = ModelRunner(config, dtype=None)
            self.scheduler = Scheduler(config)          # 池子已定 → 块数正确
            self.procs, self.broadcast_q, self.response_q = [], None, None
        else:
            # Multiproc 路径：spawn WorkerProc（每 GPU 一个）
            from engine.worker import worker_main
            bc, rc = make_channels(self.world_size)
            self.broadcast_q = [c[0] for c in bc]     # EngineCore 持 parent 端
            self.response_q = [c[0] for c in rc]
            bc_children = [c[1] for c in bc]          # child 端随 worker 传递
            rc_children = [c[1] for c in rc]
            ctx = mp.get_context("spawn")
            self.procs = []
            gpus = engine_kwargs.get("gpus") or list(range(self.world_size))
            assert len(gpus) >= self.world_size, f"GPUs {gpus} < world_size {self.world_size}"
            for r in range(self.world_size):
                # spawn 子进程 env: 只见自己的卡（worker 内 set_device(0) 即本卡）
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[r]))
                p = ctx.Process(target=_spawn_with_env,
                                args=(env, worker_main,
                                      (r, self.world_size, tp_size, ep_size,
                                       model_path, bc_children[r],
                                       rc_children[r], engine_kwargs)),
                                daemon=True)
                p.start()
                self.procs.append(p)
            # 等 ready（模型加载 + 池子分配完成，worker 回传块数）。
            # 收谁的：PP 时 stage0 的 inner-rank0（块数视图对 scheduler 有效）；
            # 非 PP 时全部 rank 都发，且块数必须一致（同预算公式）
            inner = tp_size * ep_size * dcp_size
            senders = [0] if pp_size > 1 else list(range(self.world_size))
            blocks = []
            for r in senders:
                m = self.response_q[r].recv()
                assert m.get("type") == "ready", m
                blocks.append(m["num_kvcache_blocks"])
            # TP 切分对称 → 各 rank 块数应一致；EP 下本地专家不同 → 激活峰值
            # 微差 → 块数可能差几块（实测 EP2: 1424 vs 1416）。取 min 保证
            # scheduler 的 BlockManager 不超发任何一张卡的池子。
            config.num_kvcache_blocks = min(blocks)
            self.scheduler = Scheduler(config)          # 用 worker 的真实块数
            self.inproc_runner = None

        # PP 微批流水（VirtualEngine 语义）: in-flight 微批队列。
        # 深度 = pp_size（1F1B 的最小填管道数）; 非 PP 时无流水
        self._pending_pp = []

    # ---------------- busy loop ----------------

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        msg = _spec_msg(seqs, is_prefill)

        if self.world_size == 1:
            specs = rebuild_specs(msg)
            token_ids = self.inproc_runner.run(specs, is_prefill)
        else:
            from engine.mq import broadcast
            broadcast(msg, self.broadcast_q)
            # PP 同步收口（response 从 last stage 的 inner-rank0）——
            # ⚠ 微批流水（VirtualEngine）的已知限制：完整实现需要
            # per-slot scheduler 状态（vLLM async_scheduler + virtual_engine
            # 配套），本版正确性优先保持 lockstep（bubble=(p-1)/(m+p-1) 的 m=1）
            inner = self.tp_size * self.ep_size * self.dcp_size
            resp_rank = (self.pp_size - 1) * inner if self.pp_size > 1 else 0
            out = self.response_q[resp_rank].recv()
            token_ids = out["tokens"]

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return [(s.seq_id, s.completion_token_ids) for s in seqs if s.is_finished]

    def add_request(self, prompt_ids, sampling_params=None):
        sp = sampling_params or SamplingParams(temperature=0.0)
        seq = Sequence(list(prompt_ids), sp)
        self.scheduler.add(seq)
        return seq

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self, prompts_ids, sampling_params=None):
        """批推理便捷入口（与 LLMEngine.generate 对齐）。"""
        if not isinstance(prompts_ids, list):
            prompts_ids = [prompts_ids]
        if sampling_params is None:
            from samplingParams import SamplingParams
            sampling_params = SamplingParams(temperature=0.0)
        sps = sampling_params if isinstance(sampling_params, list) \
            else [sampling_params] * len(prompts_ids)
        for p, sp in zip(prompts_ids, sps):
            self.add_request(p, sp)
        while not self.is_finished():
            self.step()
        outs = []
        return outs

    def shutdown(self):
        if self.broadcast_q is not None:
            from engine.mq import broadcast
            broadcast(SHUTDOWN, self.broadcast_q)
            for p in self.procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
