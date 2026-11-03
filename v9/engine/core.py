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

def _spawn_with_env(env, fn, args):
    """spawn 的 env 传递：mp.Process 不支持 per-child env，包一层。
    ⚠ 只覆盖目标键不 clear——spawn 的 bootstrap 依赖 mp 自己注入的环境变量，
    clear 会让子进程在 import 前死掉（zombie、无 stderr——v9.4a 实测）。"""
    os.environ.update(env)
    fn(*args)

from engine.mq import SHUTDOWN, make_channels
from engine.sequence import Sequence
from samplingParams import SamplingParams
from engine.pipeline_schedule import PipelineTopology, VirtualEnginePool


def _spec_msg(seqs, is_prefill, pipeline_lane=0, microbatch_id=0,
              virtual_engine=0):
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
    return {"type": "schedule", "scheduled": scheduled, "is_prefill": is_prefill,
            "pipeline_lane": pipeline_lane, "microbatch_id": microbatch_id,
            "virtual_engine": virtual_engine}


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
        # _spec_msg carries only this step's token window.  Preserve the
        # Sequence global indexing contract used by prepare_prefill.
        base = self.num_cached_tokens
        if isinstance(k, slice):
            start = 0 if k.start is None else max(0, k.start - base)
            stop = None if k.stop is None else max(0, k.stop - base)
            return self._tokens[start:stop:k.step]
        return self._tokens[k - base]


def rebuild_specs(msg):
    return [_Spec(m) for m in msg["scheduled"]]


class EngineCore:
    """v9.4 的引擎核心。对外暴露与 LLMEngine.generate 兼容的批推理入口。"""

    def __init__(self, model_path, tp_size=1, ep_size=1, dcp_size=1, pp_size=1,
                 master_port=29600, config_kwargs=None, gpus=None,
                 pipeline_lanes=2, pipeline_credit=4,
                 pipeline_microbatch_tokens=2048, pipeline_mode="optimized",
                 pipeline_stage_map=None):
        cpu_threads = max(1, int(os.getenv("VLLM_LEARN_CPU_THREADS", "4")))
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        self.world_size = tp_size * ep_size * dcp_size * pp_size
        self.debug = os.getenv("VLLM_LEARN_DEBUG", "0") == "1"
        self.tp_size, self.ep_size = tp_size, ep_size

        # ---- 顺序关键：runner 先于 scheduler ----
        from engine.scheduler import Scheduler
        # BlockManager 的块数来自 config.num_kvcache_blocks，那是 ModelRunner
        # warmup+分配之后才填的（v8 的 llm_engine 就是这个顺序；v9.4a 第一版
        # scheduler 在前 → BlockManager(−1→0 块) → schedule 直接 assert）
        from config import Config
        config_kwargs = dict(config_kwargs or {})
        self.config_kwargs = config_kwargs
        self.pipeline_lanes = max(1, int(pipeline_lanes))
        self.pipeline_credit = max(1, int(pipeline_credit))
        self.pipeline_microbatch_tokens = max(1, int(pipeline_microbatch_tokens))
        if pipeline_mode not in ("baseline", "optimized"):
            raise ValueError("pipeline_mode must be baseline or optimized")
        self.pipeline_mode = pipeline_mode
        self.pipeline_topology = PipelineTopology.one_to_one(pp_size)
        if pipeline_stage_map is not None:
            mapping = tuple(int(v) for v in pipeline_stage_map)
            workers = max(mapping) + 1 if mapping else 0
            candidate = PipelineTopology(pp_size, workers, mapping)
            if candidate.has_local_multiple_stages:
                raise NotImplementedError("一张卡承载多个逻辑 PP stage 需要 local-stage executor；当前 runtime 仍是一进程一 stage。")
            if mapping != tuple(range(pp_size)):
                raise NotImplementedError("当前单 stage worker runtime 只接受 identity stage_to_worker 映射。")
            self.pipeline_topology = candidate
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
            "pipeline_lanes": self.pipeline_lanes,
            "pipeline_credit": self.pipeline_credit,
            "pipeline_mode": self.pipeline_mode,
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
                # Do not rewrite CUDA_VISIBLE_DEVICES in the target wrapper:
                # spawn may import the benchmark module (and torch) before the
                # wrapper executes, which would remap every one-visible device
                # to physical GPU 0.  Keep the inherited device list and bind
                # the requested physical ordinal explicitly in worker_main.
                env = dict(os.environ)
                p = ctx.Process(target=_spawn_with_env,
                                args=(env, worker_main,
                                      (r, self.world_size, tp_size, ep_size,
                                       model_path, bc_children[r],
                                       rc_children[r], engine_kwargs,
                                       gpus[r])),
                                daemon=True)
                p.start()
                self.procs.append(p)
            # 等 ready（模型加载 + 池子分配完成，worker 回传块数）。
            # PP 必须等待每个 stage 的 inner-rank0；各 stage 层数不同，
            # scheduler 使用最小 block 数保证任一 stage 都不会超发。
            inner = tp_size * ep_size * dcp_size
            # Read one ready message from every PP stage and EP replica.
            # TP/CP ranks within a replica share the same logical KV budget;
            # EP replicas may differ by routing/activation pressure, so the
            # scheduler uses the minimum of all reported block counts.
            senders = [s * inner + e * tp_size * dcp_size
                       for s in range(pp_size) for e in range(ep_size)]
            blocks = []
            for r in senders:
                m = self.response_q[r].recv()
                if self.debug:
                    print(f"[core-ready] rank={r} blocks={m.get('num_kvcache_blocks')}", flush=True)
                assert m.get("type") == "ready", m
                blocks.append(m["num_kvcache_blocks"])
            # TP 切分对称 → 各 rank 块数应一致；EP 下本地专家不同 → 激活峰值
            # 微差 → 块数可能差几块（实测 EP2: 1424 vs 1416）。取 min 保证
            # scheduler 的 BlockManager 不超发任何一张卡的池子。
            config.num_kvcache_blocks = min(blocks)
            self.scheduler = Scheduler(config)          # 用 worker 的真实块数
            self.inproc_runner = None

        self.virtual_engine_pool = VirtualEnginePool(self.pipeline_lanes)
        self._next_microbatch_id = 0

    # ---------------- busy loop ----------------

    def _microbatches(self, seqs):
        """Create bounded PP messages while preserving sequence/KV ownership.

        A sequence is kept intact inside one message.  Scheduler chunked
        prefill handles a long request across successive steps; splitting it
        inside one step would require a second KV transaction.
        """
        if self.pp_size <= 1 or len(seqs) <= 1:
            return [seqs]
        limit = self.pipeline_microbatch_tokens
        batches, current, tokens = [], [], 0
        for seq in seqs:
            n = max(1, int(seq.num_scheduled_tokens))
            if current and tokens + n > limit:
                batches.append(current)
                current, tokens = [], 0
            current.append(seq)
            tokens += n
        if current:
            batches.append(current)
        return batches

    def step(self):
        step_start = __import__("time").perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        # AsyncEngine uses this explicit flag to distinguish an idle KV pool
        # from a completed request set.  Keeping it on EngineCore also makes
        # empty scheduling steps observable in benchmarks.
        self.last_step_had_work = bool(seqs)
        microbatches = self._microbatches(seqs)

        if self.world_size == 1:
            msg = _spec_msg(seqs, is_prefill)
            specs = rebuild_specs(msg)
            token_ids = self.inproc_runner.run(specs, is_prefill)
        else:
            from engine.mq import broadcast
            inner = self.tp_size * self.ep_size * self.dcp_size
            resp_rank = (self.pp_size - 1) * inner if self.pp_size > 1 else 0
            by_id = {}
            in_flight = 0

            def collect():
                out = self.response_q[resp_rank].recv()
                by_id.update(zip(out.get("seq_ids", []), out["tokens"]))
                self.virtual_engine_pool.complete(out.get("microbatch_id", 0))

            # One response slot plus one worker-owned result allows a window
            # of two. Drain before sending more: otherwise response ACK and
            # broadcast ACK can wait for each other indefinitely.
            window = min(2, self.pipeline_lanes)
            for mb in microbatches:
                mb_id = self._next_microbatch_id
                self._next_microbatch_id += 1
                msg = _spec_msg(
                    mb, is_prefill,
                    pipeline_lane=self.virtual_engine_pool.assign(mb_id).lane,
                    microbatch_id=mb_id,
                    virtual_engine=mb_id % self.virtual_engine_pool.size)
                msg["pipeline_mode"] = self.pipeline_mode
                broadcast(msg, self.broadcast_q)
                in_flight += 1
                if in_flight >= window:
                    collect()
                    in_flight -= 1
            for _ in range(in_flight):
                collect()
            token_ids = [by_id[s.seq_id] for s in seqs]

        # Keep a lightweight timing stream for TTFT/TPOT experiments.  It is
        # populated in the EngineCore process after worker output returns, so
        # it works for both UniProc and multiprocess TP/EP/PP paths without
        # changing the response protocol.
        step_end = __import__("time").perf_counter()
        self.last_step_events = []
        for seq, token_id in zip(seqs, token_ids):
            valid = (seq.num_cached_tokens + seq.num_scheduled_tokens
                     >= seq.num_prompt_tokens)
            self.last_step_events.append({"seq_id": seq.seq_id,
                                          "token_id": token_id,
                                          "valid_decode_token": valid,
                                          "step_start": step_start,
                                          "step_end": step_end})
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
        requested = []
        for p, sp in zip(prompts_ids, sps):
            requested.append(self.add_request(p, sp))
        while not self.is_finished():
            self.step()
        return [(seq.seq_id, list(seq.completion_token_ids))
                for seq in requested]

    def shutdown(self):
        if self.broadcast_q is not None:
            from engine.mq import broadcast
            broadcast(SHUTDOWN, self.broadcast_q)
            for p in self.procs:
                p.join(timeout=120)
                if p.is_alive():
                    p.terminate()
            for endpoint in (*self.broadcast_q, *self.response_q):
                endpoint.close()
