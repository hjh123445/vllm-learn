# v9/engine/frontend.py
"""v9.4b 前端：请求路由 + 多 EngineCore 管理（vLLM AsyncLLM 的 DP 路由形态）。

对标 vLLM v1: async_llm.py 的 AsyncLLM.generate(data_parallel_rank=...) ——
前端按 dp rank 路由请求，每个 DP rank 是一个完整的 EngineCore（独立
scheduler + KV cache）。关闭 EP 时副本独立；开启 EP 时专家池跨 DP×TP。

本版同步形态（离线批推理）: DPFrontend 持 N 个 EngineCore（每 core 内部
照常 TP/PP/DCP），add_request 按 round-robin（默认）或显式 dp_rank
分发；step 用线程并行驱动各 core（GIL 下 NCCL/CUDA 已释放，可并行）。

设计约束（vLLM 同款）:
- 请求一旦进入某个 dp rank 就不再迁移（KV cache 局部性）;
- 各 core 有独立进程、调度器和 KV cache；当前离线前端每 step 等齐结果，
  因而仍有慢副本屏障，并非完整异步服务前端;
- 前端只做聚合输出，不做调度决策（调度权在各 core 的 scheduler）。
- EP 模式统一驱动所有副本，空闲副本提交空批次，保持逐层 collective 顺序。
"""

import multiprocessing as mp
import os
import threading
import time
import traceback


def _enginecore_process(conn, model_path, core_kwargs, physical_gpus):
    """One DP replica: its own EngineCore/scheduler/KV cache/process tree.

    ``CUDA_VISIBLE_DEVICES`` is set before constructing EngineCore.  The core
    consequently receives local GPU ids ``0..N-1`` while the parent keeps the
    physical placement private to this replica.
    """
    if os.name == "posix":
        os.setsid()  # A task-owned process group, including its GPU children.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical_gpus))
    cpu_threads = max(1, int(os.getenv("VLLM_LEARN_CPU_THREADS", "4")))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    from engine.core import EngineCore
    kw = dict(core_kwargs)
    os.environ["VLLM_LEARN_DP_REPLICA_ID"] = str(kw.pop("_frontend_dp_rank", 0))
    os.environ["VLLM_LEARN_DP_REPLICA_COUNT"] = str(kw.pop("_frontend_dp_size", 1))
    kw["gpus"] = list(range(len(physical_gpus)))
    debug = os.getenv("VLLM_LEARN_DEBUG", "0") == "1"
    core = None
    try:
        core = EngineCore(model_path, **kw)
        conn.send(("ready", True))
        while True:
            cmd, payload = conn.recv()
            if debug:
                print(f"[dp-cmd] gpus={physical_gpus} cmd={cmd}", flush=True)
            if cmd == "add":
                seq = core.add_request(payload[0], payload[1])
                conn.send(("seq", seq.seq_id))
            elif cmd == "step":
                completed = core.step()
                conn.send(("step", completed, core.last_step_events))
            elif cmd == "finished":
                conn.send(("finished", core.is_finished()))
            elif cmd == "shutdown":
                core.shutdown()
                conn.send(("shutdown", True))
                break
            else:
                raise ValueError(f"unknown frontend command {cmd}")
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        finally:
            if core is not None:
                core._abort_workers()
        raise
    finally:
        conn.close()


class _DPSeq:
    __slots__ = ("seq_id",)
    def __init__(self, seq_id):
        self.seq_id = seq_id


class DPFrontend:
    """DP 前端：N 个 EngineCore + 请求路由。

    用法:
        fe = DPFrontend(model, dp_size=2, tp_size=2, ...)   # 2×TP2 = 4 GPU
        fe.add_request(ids, sp)                              # round-robin
        fe.add_request(ids, sp, dp_rank=1)                   # 显式定向
        outs = fe.run()                                      # 全部完成后聚合
    """

    def __init__(self, model_path, dp_size=1, **engine_kwargs):
        assert dp_size >= 1
        self.dp_size = dp_size
        self._closed = False
        self._shutting_down = False
        self._coupled = bool(engine_kwargs.get("enable_expert_parallel", False) and dp_size > 1)
        self._timeout = float(os.getenv("VLLM_LEARN_WORKER_TIMEOUT_S", "1200"))
        from engine.process_groups import ParallelTopology
        topology = ParallelTopology(engine_kwargs.get("tp_size", 1), dp_size,
            engine_kwargs.get("pp_size", 1), engine_kwargs.get("dcp_size", 1),
            engine_kwargs.get("enable_expert_parallel", False))
        from inspect import signature
        from engine.core import EngineCore
        unknown = set(engine_kwargs) - set(signature(EngineCore).parameters)
        if unknown:
            raise TypeError(f"Unknown EngineCore options: {sorted(unknown)}")
        if self._coupled and engine_kwargs.get("config_kwargs", {}).get("batch_execution_mode", "batched") != "batched":
            raise ValueError("Coupled DP EP cannot use per_request: collective counts would diverge")
        requested_gpus = engine_kwargs.pop("gpus", None)
        # GPU 分组: dp rank i 拿 self.gpus[i*per:(i+1)*per]
        per = topology.local_world_size
        self.gpus = requested_gpus or list(range(dp_size * per))
        assert len(self.gpus) >= dp_size * per, \
            f"GPUs {self.gpus} < dp{dp_size}×inner{per}"
        if len(set(self.gpus[:dp_size * per])) != dp_size * per:
            raise ValueError("DP workers must have distinct GPU placements")
        self.cores = []                 # process handles, kept for compatibility
        self._conns = []
        self._ctx = mp.get_context("spawn")
        self._rr = 0            # round-robin 游标
        self._seq_core = {}     # seq_id → dp_rank（输出对齐用）
        self._local_to_global = {}
        self._next_seq_id = 0
        self.last_step_events = []
        base_port = engine_kwargs.get("master_port", 29750)
        for i in range(dp_size):
            g = self.gpus[i * per:(i + 1) * per]
            kw = dict(engine_kwargs)
            kw.update(_frontend_dp_rank=i, _frontend_dp_size=dp_size)
            # 每 core 独立 rendezvous 端口（不同进程组的 TCPStore 不能撞）
            kw["master_port"] = base_port if self._coupled else base_port + i
            if self._coupled:
                kw.update(dp_size=dp_size, dp_rank=i, _dp_coordinated=True)
            parent, child = self._ctx.Pipe()
            p = self._ctx.Process(target=_enginecore_process,
                                   # EngineCore itself owns one worker process per
                                   # GPU.  A daemon DP wrapper cannot create child
                                   # processes (multiprocessing forbids nesting),
                                   # so keep the replica supervisor non-daemonic.
                                   args=(child, model_path, kw, g), daemon=False)
            p.start()
            self._conns.append(parent)
            self.cores.append(p)
        # Start EVERY replica before waiting: EP initialization spans them all.
        try:
            for i in range(dp_size):
                if self._recv(i)[0] != "ready":
                    raise RuntimeError("invalid replica startup response")
        except BaseException:
            self._abort_replicas()
            raise

    def _recv(self, index):
        conn = self._conns[index]
        if hasattr(conn, "poll"):
            deadline = time.monotonic() + self._timeout
            while not conn.poll(.1):
                observed = [self.cores[index]] if self._shutting_down else self.cores
                if any(not p.is_alive() for p in observed):
                    raise RuntimeError("DP replica exited before replying")
                if time.monotonic() >= deadline:
                    raise TimeoutError("DP replica response timed out")
        response = conn.recv()
        if response[0] == "error":
            raise RuntimeError(response[1])
        return response

    def _abort_replicas(self):
        self._closed = True
        import signal
        for process in self.cores:
            if os.name == "posix":
                try:
                    if os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if process.is_alive():
                process.terminate()
        for process in self.cores:
            process.join(timeout=2)
        for conn in self._conns:
            conn.close()

    # ---------------- 请求路由 ----------------

    def add_request(self, prompt_ids, sampling_params=None, dp_rank=None):
        """dp_rank=None → round-robin（负载均匀的默认）；显式指定则定向
        （vLLM 的 data_parallel_rank 参数同款语义——调用方按业务粘性路由）。"""
        if dp_rank is None:
            dp_rank = self._rr
            self._rr = (self._rr + 1) % self.dp_size
        if not 0 <= dp_rank < self.dp_size:
            raise ValueError(f"invalid dp_rank {dp_rank}")
        conn = self._conns[dp_rank]
        conn.send(("add", (prompt_ids, sampling_params)))
        _, local_id = self._recv(dp_rank)
        # Each spawned EngineCore starts its own Sequence counter at zero.
        # Namespace local ids before merging outputs from different replicas.
        seq = _DPSeq(self._next_seq_id)
        self._next_seq_id += 1
        self._local_to_global[dp_rank, local_id] = seq.seq_id
        self._seq_core[seq.seq_id] = dp_rank
        return seq, dp_rank

    # ---------------- 驱动 ----------------

    def step_all(self):
        """并发驱动各进程一步，等齐后聚合；异常传播至调用线程。"""
        results = {}
        errors = []
        self.last_step_events = []
        coupled = getattr(self, "_coupled", False)
        states = None
        if coupled:
            for conn in self._conns:
                conn.send(("finished", None))
            states = [self._recv(i)[1] for i in range(self.dp_size)]
            if all(states):
                return {}

        def _drive(i):
            try:
                _step(i)
            except Exception as exc:
                errors.append(exc)

        def _step(i):
            conn = self._conns[i]
            if coupled:
                finished = states[i]
            else:
                conn.send(("finished", None))
                _, finished = self._recv(i)
            if coupled or not finished:
                conn.send(("step", None))
                reply = self._recv(i)
                step_out = reply[1]
                events = reply[2] if len(reply) > 2 else []
                for event in events:
                    global_id = self._local_to_global[i, event["seq_id"]]
                    self.last_step_events.append(dict(
                        event, seq_id=global_id, dp_rank=i))
                for seq_id, toks in step_out:
                    global_id = self._local_to_global.pop((i, seq_id))
                    self._seq_core.pop(global_id, None)
                    results[global_id] = toks

        threads = [threading.Thread(target=_drive, args=(i,))
                   for i in range(self.dp_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            if coupled:
                self._abort_replicas()
            raise RuntimeError("DP replica step failed") from errors[0]
        return results

    def run(self):
        """跑完所有在途请求，返回 {seq_id: completion_tokens}。"""
        outs = {}
        while True:
            states = []
            for conn in self._conns:
                conn.send(("finished", None))
            for i in range(self.dp_size):
                states.append(self._recv(i)[1])
            if all(states):
                break
            outs.update(self.step_all())
        return outs

    def shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._shutting_down = True
        try:
            for conn in self._conns:
                conn.send(("shutdown", None))
            for i in range(self.dp_size):
                self._recv(i)
            for process in self.cores:
                process.join(timeout=10)
        finally:
            self._abort_replicas()
