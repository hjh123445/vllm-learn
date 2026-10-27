# v5/nanovllm/engine/async_engine.py
"""v5 在线推理引擎：LLMEngine + 后台循环线程 + 每请求的流式队列。

线程模型（单卡单进程内）：
  - HTTP 处理线程：tokenize → async_engine.add() → 拿到 queue → 从 queue 逐个取 token
  - 引擎循环线程（唯一写者）：被 wake 唤醒 → schedule → run → postprocess →
    把每条序列的新 token 推进它的 queue
线程安全依据：scheduler 的 waiting 是 deque（append/popleft 在 CPython 里原子，
见 benchmark/PERFORMANCE_NOTES §0 的讨论）；block_manager 和 Sequence 状态只有
循环线程写；add() 只往 waiting 投放数据——单写多读无竞态。"""
import queue
import threading
import time


class _StreamState:
    """一条在线请求的流式状态（引擎循环线程写，HTTP 线程读）。"""
    __slots__ = ("q", "sent", "finished", "finish_reason", "error")

    def __init__(self):
        self.q = queue.Queue()
        self.sent = 0              # 已推送的 token 数
        self.finished = False
        self.finish_reason = None
        self.error = None

    def close(self):
        self.q.put(None)            # 结束哨兵


class AsyncEngine:

    def __init__(self, model: str, **kwargs):
        from nanovllm.engine.llm_engine import LLMEngine
        self.engine = LLMEngine(model, **kwargs)
        self.states: dict[int, _StreamState] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    # ============ HTTP 线程调用 ============

    def add(self, prompt: str | list[int], sampling_params) -> tuple[int, _StreamState]:
        """提交一条请求，立刻返回 (请求 id, 流队列)。queue 里逐个出 token，None 表示结束。"""
        seq = self.engine.add_request(prompt, sampling_params)
        st = _StreamState()
        self.states[seq.seq_id] = st
        self._wake.set()                     # 唤醒引擎循环
        return seq.seq_id, st

    def start(self):
        assert self._thread is None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ============ 引擎循环线程 ============

    def _loop(self):
        while not self._stop.is_set():
            if self.engine.is_finished():
                self._wake.wait(timeout=0.005)    # 空转时让出 CPU，add() 会唤醒
                self._wake.clear()
                continue
            seqs, is_prefill = self.engine.scheduler.schedule()
            if not seqs:
                continue
            tokens = self.engine.model_runner.run(seqs, is_prefill)
            self.engine.scheduler.postprocess(seqs, tokens, is_prefill)
            # 流式推送：本步每条序列新产生的 token
            for seq in seqs:
                st = self.states.get(seq.seq_id)
                if st is None:
                    continue
                comp = seq.completion_token_ids
                while st.sent < len(comp):
                    st.q.put(comp[st.sent])
                    st.sent += 1
                if seq.is_finished and not st.finished:
                    st.finished = True
                    st.finish_reason = seq.status.name.lower()
                    st.close()

    # ============ 便捷：一次性非流式 ============

    def generate(self, prompt, sampling_params):
        _, st = self.add(prompt, sampling_params)
        toks = []
        while True:
            t = st.q.get()
            if t is None:
                return toks, st.finish_reason
            toks.append(t)
