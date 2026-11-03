# v9/engine/frontend.py
"""v9.4b 前端：请求路由 + 多 EngineCore 管理（vLLM AsyncLLM 的 DP 路由形态）。

对标 vLLM v1: async_llm.py 的 AsyncLLM.generate(data_parallel_rank=...) ——
前端按 dp rank 路由请求，每个 DP rank 是一个完整的 EngineCore（独立
scheduler + KV cache + 完整模型副本），DP rank 之间零通信。

本版同步形态（离线批推理）: DPFrontend 持 N 个 EngineCore（每 core 内部
照常 TP/EP/PP/DCP），add_request 按 round-robin（默认）或显式 dp_rank
分发；step 用线程并行驱动各 core（GIL 下 NCCL/CUDA 已释放，可并行）。

设计约束（vLLM 同款）:
- 请求一旦进入某个 dp rank 就不再迁移（KV cache 局部性）;
- 各 core 有独立进程、调度器和 KV cache；当前离线前端每 step 等齐结果，
  因而仍有慢副本屏障，并非完整异步服务前端;
- 前端只做聚合输出，不做调度决策（调度权在各 core 的 scheduler）。
"""

import multiprocessing as mp
import os
import threading


def _enginecore_process(conn, model_path, core_kwargs, physical_gpus):
    """One DP replica: its own EngineCore/scheduler/KV cache/process tree.

    ``CUDA_VISIBLE_DEVICES`` is set before constructing EngineCore.  The core
    consequently receives local GPU ids ``0..N-1`` while the parent keeps the
    physical placement private to this replica.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical_gpus))
    cpu_threads = max(1, int(os.getenv("VLLM_LEARN_CPU_THREADS", "4")))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    from engine.core import EngineCore
    kw = dict(core_kwargs)
    kw["gpus"] = list(range(len(physical_gpus)))
    debug = os.getenv("VLLM_LEARN_DEBUG", "0") == "1"
    core = EngineCore(model_path, **kw)
    if debug:
        print(f"[dp-core-ready] gpus={physical_gpus}", flush=True)
    while True:
        cmd, payload = conn.recv()
        if debug:
            print(f"[dp-cmd] gpus={physical_gpus} cmd={cmd}", flush=True)
        if cmd == "add":
            seq = core.add_request(payload[0], payload[1])
            conn.send(("seq", seq.seq_id))
        elif cmd == "step":
            conn.send(("step", core.step()))
        elif cmd == "finished":
            conn.send(("finished", core.is_finished()))
        elif cmd == "shutdown":
            core.shutdown()
            conn.send(("shutdown", True))
            break


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
        requested_gpus = engine_kwargs.pop("gpus", None)
        # GPU 分组: dp rank i 拿 self.gpus[i*per:(i+1)*per]
        per = (engine_kwargs.get("tp_size", 1) * engine_kwargs.get("ep_size", 1)
               * engine_kwargs.get("dcp_size", 1) * engine_kwargs.get("pp_size", 1))
        self.gpus = requested_gpus or list(range(dp_size * per))
        assert len(self.gpus) >= dp_size * per, \
            f"GPUs {self.gpus} < dp{dp_size}×inner{per}"
        self.cores = []                 # process handles, kept for compatibility
        self._conns = []
        self._ctx = mp.get_context("spawn")
        self._rr = 0            # round-robin 游标
        self._seq_core = {}     # seq_id → dp_rank（输出对齐用）
        self._local_to_global = {}
        self._next_seq_id = 0
        base_port = engine_kwargs.get("master_port", 29750)
        for i in range(dp_size):
            g = self.gpus[i * per:(i + 1) * per]
            kw = dict(engine_kwargs)
            # 每 core 独立 rendezvous 端口（不同进程组的 TCPStore 不能撞）
            kw["master_port"] = base_port + i
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
        _, local_id = conn.recv()
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

        def _drive(i):
            try:
                _step(i)
            except Exception as exc:
                errors.append(exc)

        def _step(i):
            conn = self._conns[i]
            conn.send(("finished", None))
            _, finished = conn.recv()
            if not finished:
                conn.send(("step", None))
                _, step_out = conn.recv()
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
            raise RuntimeError("DP replica step failed") from errors[0]
        return results

    def run(self):
        """跑完所有在途请求，返回 {seq_id: completion_tokens}。"""
        outs = {}
        while True:
            states = []
            for conn in self._conns:
                conn.send(("finished", None))
            for conn in self._conns:
                states.append(conn.recv()[1])
            if all(states):
                break
            outs.update(self.step_all())
        return outs

    def shutdown(self):
        for conn in self._conns:
            conn.send(("shutdown", None))
        for conn in self._conns:
            conn.recv()
        for p in self.cores:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
