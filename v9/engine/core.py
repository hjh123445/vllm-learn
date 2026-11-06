# v9/engine/core.py
"""v9.4 EngineCore：scheduler + executor 同进程（vLLM v1 形态的核心）。

对标 vLLM: v1/engine/core.py 的 EngineCore（scheduler 唯一决策者）+
v1/executor/multiproc_executor.py 的 MultiprocExecutor（spawn WorkerProc）。

PP busy loop: one scheduler -> non-blocking executor -> persistent batch queue.
Fill the bounded queue first; settle the oldest result when full or when no
independent requests are ready. No iteration-wide microbatch drain/barrier.

单卡退化（对齐 UniProcExecutor）: world_size==1 时不 spawn 子进程，
ModelRunner 同进程直调——同一条 EngineCore 代码路径。

"""

import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from time import perf_counter

def _spawn_with_env(env, fn, args):
    """spawn 的 env 传递：mp.Process 不支持 per-child env，包一层。
    ⚠ 只覆盖目标键不 clear——spawn 的 bootstrap 依赖 mp 自己注入的环境变量，
    clear 会让子进程在 import 前死掉（zombie、无 stderr——v9.4a 实测）。"""
    os.environ.update(env)
    fn(*args)

from engine.mq import make_channels
from engine.sequence import Sequence
from samplingParams import SamplingParams
from engine.process_groups import ParallelTopology


def _spec_msg(seqs, is_prefill, microbatch_id=0):
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
            "block_size": getattr(s, "block_size", 256),
        })
    return {"type": "schedule", "scheduled": scheduled, "is_prefill": is_prefill,
            "microbatch_id": microbatch_id}


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
        bs = self.block_size = m.get("block_size", 256)
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


@dataclass
class _PendingBatch:
    batch_id: int
    seqs: tuple
    is_prefill: bool
    submitted_at: float
    future: Future


class EngineCore:
    """v9.4 的引擎核心。对外暴露与 LLMEngine.generate 兼容的批推理入口。"""

    def __init__(self, model_path, tp_size=1, dcp_size=1, pp_size=1,
                 master_port=29600, config_kwargs=None, gpus=None,
                 pipeline_credit=4,
                 pipeline_microbatch_tokens=2048, pipeline_mode="optimized",
                 max_concurrent_batches=None,
                 enable_expert_parallel=False, dp_size=1, dp_rank=0,
                 _dp_coordinated=False, runner_cls=None, transport_cls=None):
        self._closed = False
        self._worker_timeout = float(os.getenv("VLLM_LEARN_WORKER_TIMEOUT_S", "1200"))
        if self._worker_timeout <= 0:
            raise ValueError("worker timeout must be positive")
        cpu_threads = max(1, int(os.getenv("VLLM_LEARN_CPU_THREADS", "4")))
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        topology = ParallelTopology(tp_size, dp_size, pp_size, dcp_size,
                                    enable_expert_parallel)
        if not 0 <= dp_rank < dp_size:
            raise ValueError("invalid dp_rank")
        if dp_size > 1 and not _dp_coordinated:
            raise ValueError("Multi-DP EP requires DPFrontend to coordinate idle replicas")
        self.world_size = topology.local_world_size
        self.global_world_size = topology.world_size
        self._coupled_dp = enable_expert_parallel and dp_size > 1
        self.debug = os.getenv("VLLM_LEARN_DEBUG", "0") == "1"

        # ---- 顺序关键：runner 先于 scheduler ----
        from engine.scheduler import Scheduler
        # BlockManager 的块数来自 config.num_kvcache_blocks，那是 ModelRunner
        # warmup+分配之后才填的（v8 的 llm_engine 就是这个顺序；v9.4a 第一版
        # scheduler 在前 → BlockManager(−1→0 块) → schedule 直接 assert）
        from config import Config
        config_kwargs = dict(config_kwargs or {})
        if self._coupled_dp and config_kwargs.get("batch_execution_mode", "batched") != "batched":
            raise ValueError("Coupled DP EP requires batched execution; per_request would issue unequal collective counts")
        if pipeline_credit < 1:
            raise ValueError("pipeline_credit must be positive")
        self.pipeline_microbatch_tokens = max(1, int(pipeline_microbatch_tokens))
        self.max_concurrent_batches = (pp_size if max_concurrent_batches is None
                                       else int(max_concurrent_batches))
        if self.max_concurrent_batches < 1:
            raise ValueError("max_concurrent_batches must be positive")
        if pipeline_mode not in ("baseline", "optimized"):
            raise ValueError("pipeline_mode must be baseline or optimized")
        config = Config(model_path, **config_kwargs)
        self.config = config
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        eos = self.tokenizer.eos_token_id
        config.eos = eos[0] if isinstance(eos, list) else eos

        engine_kwargs = {
            "master_addr": "127.0.0.1", "master_port": master_port,
            "config_kwargs": config_kwargs, "topology": topology, "dp_rank": dp_rank,
            "pipeline_credit": pipeline_credit, "pipeline_mode": pipeline_mode,
            # Optional importable implementations, preserved across spawn.
            "runner_cls": runner_cls, "transport_cls": transport_cls,
        }
        self.pp_size = pp_size

        if self.global_world_size == 1:
            # UniProc 路径：ModelRunner 同进程直调（分配池子 → 填 config）
            from engine.model_runner import ModelRunner
            self.inproc_runner = (runner_cls or ModelRunner)(config, dtype=None)
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
            gpus = gpus or list(range(self.world_size))
            assert len(gpus) >= self.world_size, f"GPUs {gpus} < world_size {self.world_size}"
            if len(set(gpus[:self.world_size])) != self.world_size:
                raise ValueError("GPU workers require distinct device ordinals")
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
                                      (r, model_path, bc_children[r],
                                       rc_children[r], engine_kwargs,
                                       gpus[r])),
                                daemon=True)
                p.start()
                self.procs.append(p)
            # 等 ready（模型加载 + 池子分配完成，worker 回传块数）。
            # PP 必须等待每个 stage 的 inner-rank0；各 stage 层数不同，
            # scheduler 使用最小 block 数保证任一 stage 都不会超发。
            inner = tp_size * dcp_size
            # Read one ready message from every PP stage and EP replica.
            # TP/CP ranks within a replica share the same logical KV budget;
            # EP replicas may differ by routing/activation pressure, so the
            # scheduler uses the minimum of all reported block counts.
            senders = [s * inner for s in range(pp_size)]
            blocks = []
            for r in senders:
                self._wait_for_worker(self.response_q[r].notify)
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

        self.executor = None
        if self.global_world_size > 1:
            from engine.executor import MultiprocessExecutor
            inner = tp_size * dcp_size
            self.executor = MultiprocessExecutor(
                self.broadcast_q, self.response_q[(pp_size - 1) * inner],
                self.procs, self._worker_timeout, self.max_concurrent_batches)
        self._initialize_batch_queue()

    def _initialize_batch_queue(self):
        self.batch_queue = deque()
        self._next_microbatch_id = 0
        self.last_step_events = []
        self.last_step_had_work = False

    # ---------------- busy loop ----------------

    def _schedule_batch(self):
        if self.pp_size <= 1:
            return self.scheduler.schedule()
        # Split decode requests too: a token-only budget would put every
        # small decode into one batch and leave the remaining PP stages idle.
        # This is a dynamic per-batch cap, not P fixed request partitions.
        requests = len(self.scheduler.running) + len(self.scheduler.waiting)
        width = min(requests, self.scheduler.max_num_seqs)
        max_seqs = max(1, (width + self.max_concurrent_batches - 1)
                       // self.max_concurrent_batches)
        return self.scheduler.schedule(
            token_budget=self.pipeline_microbatch_tokens, max_seqs=max_seqs)

    def _submit_batch(self, seqs, is_prefill):
        batch_id = self._next_microbatch_id
        self._next_microbatch_id += 1
        message = _spec_msg(seqs, is_prefill, microbatch_id=batch_id)
        submitted = perf_counter()
        if self.executor is None:
            future = Future()
            try:
                tokens = self.inproc_runner.run(rebuild_specs(message), is_prefill,
                                               microbatch_id=batch_id)
                future.set_result({"type": "output", "tokens": tokens,
                                   "seq_ids": [s.seq_id for s in seqs],
                                   "microbatch_id": batch_id})
            except Exception as error:
                future.set_exception(error)
        else:
            future = self.executor.execute_model(message)
        self.batch_queue.append(_PendingBatch(
            batch_id, tuple(seqs), is_prefill, submitted, future))

    def _complete_batch(self):
        batch = self.batch_queue[0]
        output = batch.future.result(timeout=self._worker_timeout)
        seqs = batch.seqs
        if (output.get("microbatch_id") != batch.batch_id
                or output.get("seq_ids") != [s.seq_id for s in seqs]
                or len(output.get("tokens", ())) != len(seqs)):
            raise RuntimeError(f"output does not match batch {batch.batch_id}")
        token_ids = output["tokens"]
        completed_at = perf_counter()
        cancelled = {s.seq_id for s in seqs if self.scheduler.is_cancelled(s.seq_id)}
        self.last_step_events = [
            {"seq_id": seq.seq_id, "token_id": token,
             "valid_decode_token": (seq.num_cached_tokens + seq.num_scheduled_tokens
                                    >= seq.num_prompt_tokens),
             "step_start": batch.submitted_at, "step_end": completed_at,
             "batch_id": batch.batch_id}
            for seq, token in zip(seqs, token_ids) if seq.seq_id not in cancelled]
        self.scheduler.postprocess(seqs, token_ids, batch.is_prefill)
        self.batch_queue.popleft()
        return [(s.seq_id, s.completion_token_ids) for s in seqs
                if s.is_finished and s.seq_id not in cancelled]

    def step(self):
        """Prefer submission; settle one oldest batch when full/no ready work.

        A call may return no tokens while filling the pipeline. Outstanding
        batches survive this call and protect their requests/KV allocations.
        """
        if self._closed:
            raise RuntimeError("EngineCore is closed")
        self.last_step_events = []
        self.last_step_had_work = bool(self.batch_queue)
        try:
            if len(self.batch_queue) < self.max_concurrent_batches:
                seqs, is_prefill = self._schedule_batch()
                if seqs or getattr(self, "_coupled_dp", False):
                    self._submit_batch(seqs, is_prefill)
                    self.last_step_had_work = True
                    if len(self.batch_queue) < self.max_concurrent_batches:
                        return []
            if self.batch_queue:
                return self._complete_batch()
            return []
        except Exception:
            self._abort_workers()
            raise

    def add_request(self, prompt_ids, sampling_params=None):
        sp = sampling_params or SamplingParams(temperature=0.0)
        seq = Sequence(list(prompt_ids), sp)
        seq.block_size = self.config.kvcache_block_size
        self.scheduler.add(seq)
        return seq

    def is_finished(self):
        # Dummy EP participation can remain in flight after this replica's
        # requests finish. The frontend stops globally, then drains all queues.
        pending_requests = any(batch.seqs for batch in self.batch_queue)
        return not pending_requests and self.scheduler.is_finished()

    def abort_request(self, seq_id):
        """Cancel immediately; defer KV reclamation until in-flight work ends."""
        return self.scheduler.abort(seq_id)

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

    def _wait_for_worker(self, connection):
        from engine.worker_health import wait_for_workers
        try:
            wait_for_workers(connection, self.procs, self._worker_timeout)
        except (RuntimeError, TimeoutError):
            self._abort_workers()
            raise

    def _abort_workers(self):
        """Release only this EngineCore's children and shared-memory slots."""
        self._closed = True
        executor = getattr(self, "executor", None)
        try:
            if executor is not None:
                executor.abort()
        finally:
            for p in self.procs:
                if p.is_alive():
                    p.terminate()
            for p in self.procs:
                p.join(timeout=2)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)
            for endpoint in (*(self.broadcast_q or []), *(self.response_q or [])):
                endpoint.close()

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        try:
            # Stop scheduling; retire/cancel the outstanding work before
            # asking workers to exit, so no activation recv is orphaned.
            for seq in (*self.scheduler.waiting, *self.scheduler.running):
                self.scheduler.abort(seq.seq_id)
            while self.batch_queue:
                self._complete_batch()
            if self.executor is not None:
                self.executor.shutdown()
                for p in self.procs:
                    p.join(timeout=5)
            runner = getattr(self, "inproc_runner", None)
            if runner is not None:
                runner.shutdown()
        finally:
            if self.broadcast_q is not None:
                self._abort_workers()
            self._closed = True
