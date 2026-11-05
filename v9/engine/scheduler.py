from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: "Config"):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,
                                          config.enable_prefix_caching)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # One outstanding batch per request. KV remains owned until its
        # matching output is settled, including cancellation and preemption.
        self.in_flight: set[int] = set()
        self.cancelled: set[int] = set()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self, *, token_budget=None, max_seqs=None) -> tuple[list[Sequence], bool]:
        # ---- vLLM v1 语义（Sarathi-Serve 混合批次）: 单循环, 无 prefill/decode 之分 ----
        # 每个请求只有 num_computed_tokens 和 num_tokens, scheduler 的工作是
        # 在 token_budget 内给各请求分配 token 让 computed 追上 total——
        # 任意请求可被 chunk（预算装不下就吃掉剩余预算）, RUNNING 优先于
        # WAITING（喂饱在途的再收新的）, decode(1 token)与 prefill 尾段/首段
        # 混入同一批（消相位切换——v0 '整批 prefill 时 decode 干等'的 TPOT
        # 尖刺来源）。
        token_budget = min(self.max_num_batched_tokens,
                           self.max_num_batched_tokens if token_budget is None else token_budget)
        batch_limit = self.max_num_seqs if max_seqs is None else min(max_seqs, self.max_num_seqs)
        if token_budget < 1 or batch_limit < 1:
            raise ValueError("batch token and request budgets must be positive")
        scheduled_seqs = []

        def _try_schedule(seq, in_waiting):
            nonlocal token_budget
            if seq.seq_id in self.in_flight:
                return False
            # APC 命中检查（仅无 block_table 的新请求）
            num_cached_blocks = 0
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    return False                      # 池子满: 触发抢占的信号
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 追赶语义: 本步给 min(缺的, 预算剩的) —— chunk 自然发生
            num_sched = min(num_tokens, token_budget)
            if num_sched == 0:
                return False
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            # 块预算检查（KV 不够则 preempt running 尾部, vLLM 同款）
            while not self.block_manager.can_append_chunk(seq, num_sched):
                # A previously scheduled request already owns this step's KV
                # transaction. Never free/reuse those blocks before forward.
                victim = next((s for s in reversed(self.running)
                               if s is not seq and s.seq_id not in self.in_flight
                               and s not in scheduled_seqs), None)
                if victim is not None:
                    self.preempt(victim)
                else:
                    self.preempt(seq)
                    return False
            self.block_manager.append_chunk(seq, num_sched)
            seq.num_scheduled_tokens = num_sched
            token_budget -= num_sched
            if in_waiting:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            scheduled_seqs.append(seq)
            return True

        # 第一遍: RUNNING（chunked 尾段 + decode 混跑）
        for seq in list(self.running):
            if seq not in self.running:
                continue
            if token_budget <= 0 or len(scheduled_seqs) >= batch_limit:
                break
            _try_schedule(seq, in_waiting=False)

        # 第二遍: WAITING（新 prefill, 与 RUNNING 的 decode 混批）
        while (self.waiting and token_budget > 0
               and len(scheduled_seqs) < batch_limit
               and len(self.running) < self.max_num_seqs):
            if not _try_schedule(self.waiting[0], in_waiting=True):
                break

        # A full KV pool can temporarily make every request unschedulable.
        # Returning an empty scheduling result lets the executor apply
        # backpressure instead of crashing on an assertion (or dereferencing
        # an empty batch).  vLLM's scheduler uses the same explicit empty
        # output protocol.
        if not scheduled_seqs:
            return [], False
        # 混合批次标记: 只要还有任何请求没跑完 prompt 就算"含 prefill"
        # （runner 用 varlen 形态; 纯 decode 全 1-token 批走 decode 快路径）
        has_prefill = any(s.num_cached_tokens < s.num_prompt_tokens for s in scheduled_seqs)
        self.in_flight.update(s.seq_id for s in scheduled_seqs)
        return scheduled_seqs, has_prefill

    def preempt(self, seq: Sequence):
        if seq.seq_id in self.in_flight:
            raise RuntimeError("Cannot preempt a request with outstanding GPU work")
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0
        self.waiting.appendleft(seq)

    def abort(self, seq_id):
        for pending in (self.waiting, self.running):
            for seq in list(pending):
                if seq.seq_id != seq_id:
                    continue
                if seq_id in self.in_flight:
                    self.cancelled.add(seq_id)
                else:
                    self._finish(seq)
                return True
        return False

    def _finish(self, seq):
        self.block_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0
        seq.status = SequenceStatus.FINISHED
        for pending in (self.waiting, self.running):
            if seq in pending:
                pending.remove(seq)

    def is_cancelled(self, seq_id):
        return seq_id in self.cancelled

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        if len(token_ids) != len(seqs):
            raise ValueError(f"scheduler/output length mismatch: {len(seqs)} != {len(token_ids)}")
        if any(s.seq_id not in self.in_flight for s in seqs):
            raise RuntimeError("Output does not own an outstanding request reservation")
        # v9.4e 混合批次: is_prefill 只是"本批含 prefill 段"的标记——
        # append_token 的判定【逐序列】做: 该序列的 prompt 已完整算完
        # (num_cached == prompt 长度) 才消费采样 token; chunk 中的序列
        # 本步的采样 token 无意义, 丢弃
        for seq, token_id in zip(seqs, token_ids):
            self.in_flight.remove(seq.seq_id)
            if seq.seq_id in self.cancelled:
                self.cancelled.remove(seq.seq_id)
                self._finish(seq)
                continue
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                seq.is_prefill = True
                continue                                  # chunk 未完: 不 append
            seq.is_prefill = False
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                self._finish(seq)
