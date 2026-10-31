from collections import deque

from config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---- vLLM v1 语义（Sarathi-Serve 混合批次）: 单循环, 无 prefill/decode 之分 ----
        # 每个请求只有 num_computed_tokens 和 num_tokens, scheduler 的工作是
        # 在 token_budget 内给各请求分配 token 让 computed 追上 total——
        # 任意请求可被 chunk（预算装不下就吃掉剩余预算）, RUNNING 优先于
        # WAITING（喂饱在途的再收新的）, decode(1 token)与 prefill 尾段/首段
        # 混入同一批（消相位切换——v0 '整批 prefill 时 decode 干等'的 TPOT
        # 尖刺来源）。
        token_budget = self.max_num_batched_tokens
        scheduled_seqs = []

        def _try_schedule(seq, in_waiting):
            nonlocal token_budget
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
                if self.running:
                    self.preempt(self.running.pop())
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
            if token_budget <= 0:
                break
            _try_schedule(seq, in_waiting=False)

        # 第二遍: WAITING（新 prefill, 与 RUNNING 的 decode 混批）
        while self.waiting and token_budget > 0 and len(self.running) < self.max_num_seqs:
            if not _try_schedule(self.waiting[0], in_waiting=True):
                break

        assert scheduled_seqs
        # 混合批次标记: 只要还有任何请求没跑完 prompt 就算"含 prefill"
        # （runner 用 varlen 形态; 纯 decode 全 1-token 批走 decode 快路径）
        has_prefill = any(s.num_cached_tokens < s.num_prompt_tokens for s in scheduled_seqs)
        return scheduled_seqs, has_prefill

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # v9.4e 混合批次: is_prefill 只是"本批含 prefill 段"的标记——
        # append_token 的判定【逐序列】做: 该序列的 prompt 已完整算完
        # (num_cached == prompt 长度) 才消费采样 token; chunk 中的序列
        # 本步的采样 token 无意义, 丢弃
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                continue                                  # chunk 未完: 不 append
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
