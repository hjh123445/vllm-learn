# v5/engine/async_engine.py
"""v5 在线推理引擎：LLMEngine + 后台循环线程 + 每请求的流式队列。

线程模型（单卡单进程内）：
  - HTTP 处理线程：tokenize → async_engine.add() → 拿到 queue → 从 queue 逐个取 token
  - 引擎循环线程（唯一写者）：被 wake 唤醒 → schedule → run → postprocess →
    把每条序列的新 token 推进它的 queue
线程安全依据：scheduler 的 waiting 是 deque（append/popleft 在 CPython 里原子，
见 benchmark/PERFORMANCE_NOTES §0 的讨论）；block_manager 和 Sequence 状态只有
循环线程写；add() 只往 waiting 投放数据——单写多读无竞态。

P0-6 修复（引擎死亡路径）：_loop 整体包 try/except——任何异常（OOM/shape 错/
scheduler assert）不再让 daemon 线程静默死亡、全部请求无 timeout 挂死，而是
给每个在途 state 置 error 并 close（HTTP 线程拿到哨兵后读 st.error 返回 500），
并置 _dead 让后续 add() 快速失败。对齐 vLLM 的 ENGINE_CORE_DEAD 语义（其
EngineCore 死亡 → 前端 output_handler → propagate_error → 请求级异常）。

P0-6b 修复（发布竞态）：add() 先注册 states 再入队——旧版先 add_request 后
注册 state，引擎线程在两步之间完成整个请求（max_tokens=1 或首 token EOS）时，
close 哨兵发向不存在的 state，HTTP 线程永久阻塞。state 先于请求可见，竞态归零。

P0-6c 修复（states 泄漏 + 空转）：finish 推送后 del self.states[seq_id]——
旧版只增不删，长服务下每请求泄漏一个 _StreamState；空批分支加 sleep 防
池满场景的 100% CPU 热自旋。"""
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
        from engine.llm_engine import LLMEngine
        self.engine = LLMEngine(model, **kwargs)
        self.states: dict[int, _StreamState] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._dead: BaseException | None = None      # 引擎死亡原因（P0-6）

    # ============ HTTP 线程调用 ============

    def add(self, prompt: str | list[int], sampling_params) -> tuple[int, _StreamState]:
        """提交一条请求，立刻返回 (请求 id, 流队列)。queue 里逐个出 token，None 表示结束。

        顺序关键（P0-6b）：先注册 state 再入队——引擎线程任何时刻 schedule 到
        这条请求都能在 states 里找到它的队列；先 add_request 的话，请求可能在
        state 注册前就完成（max_tokens=1 / 首 token EOS），结束哨兵发向不存在
        的 state，调用方永久挂死。"""
        if self._dead is not None:
            raise RuntimeError(f"引擎已死亡: {self._dead!r}") from self._dead
        seq_id = self.engine.next_seq_id()            # 预读号（不消费计数器）
        st = _StreamState()
        self.states[seq_id] = st                      # state 先于请求可见
        seq = self.engine.add_request(prompt, sampling_params, seq_id=seq_id)
        assert seq.seq_id == seq_id
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
        # P0-6：引擎线程的任何异常都是全局故障——广播给所有在途请求，
        # 而不是静默死亡让它们挂死在 queue.get() 上
        try:
            self._loop_inner()
        except BaseException as e:                    # noqa: BLE001 —— 广播后重抛
            self._dead = e
            for st in self.states.values():
                if not st.finished:
                    st.error = e
                    st.finished = True
                    st.finish_reason = "error"
                    st.close()
            raise

    def _loop_inner(self):
        while not self._stop.is_set():
            if self.engine.is_finished():
                self._wake.wait(timeout=0.005)    # 空转时让出 CPU，add() 会唤醒
                self._wake.clear()
                continue
            seqs, is_prefill = self.engine.scheduler.schedule()
            if not seqs:
                # 空批防御（P0-6c）：池满/准入失败时避免 100% CPU 热自旋
                time.sleep(0.001)
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
                    # P0-6c：finish 后清理 state——HTTP 线程拿到哨兵即结束，
                    # finish_reason 已在哨兵前写好；del 防止 states 只增不减
                    del self.states[seq.seq_id]

    # ============ 便捷：一次性非流式 ============

    def generate(self, prompt, sampling_params):
        _, st = self.add(prompt, sampling_params)
        toks = []
        while True:
            t = st.q.get()
            if t is None:
                if st.error is not None:
                    raise RuntimeError(f"引擎异常: {st.error!r}") from st.error
                return toks, st.finish_reason
            toks.append(t)
