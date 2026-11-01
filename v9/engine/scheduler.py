from collections import deque

from config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = getattr(config, "max_model_len", None)
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 准入校验（P0 配套）：放不进池子/超出 max_model_len 的请求在入队期
        # 拒绝（对齐 vLLM kv_cache_utils._check_enough_kv_cache_memory +
        # 入队期校验），而不是让 schedule() 落到 `assert scheduled_seqs` 崩溃
        # 或 v2 式空转死循环。保守判据（忽略 APC 命中）：宁可拒绝也不死锁。
        max_total = len(seq) + seq.max_tokens
        if self.max_model_len is not None and max_total > self.max_model_len:
            raise ValueError(
                f"请求过长: prompt {len(seq)} + max_tokens {seq.max_tokens} = "
                f"{max_total} > max_model_len {self.max_model_len}（拒绝准入）")
        need_blocks = (max_total + self.block_size - 1) // self.block_size
        total_blocks = len(self.block_manager.blocks)
        if need_blocks > total_blocks:
            raise ValueError(
                f"请求超出 KV 池容量: 需 {need_blocks} blocks（prompt {len(seq)} + "
                f"max_tokens {seq.max_tokens} @ block_size {self.block_size}），"
                f"池共 {total_blocks} blocks——增大 gpu_memory_utilization 或减小请求")
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
            # P0-16 修复：两种"自己被抢占"的形态都必须退出本轮——被抢占后
            # num_cached_tokens 清 0 / 块表清空，而 num_sched 是按【抢占前】
            # 的 frontier 算的（陈旧）；decode 路径的 context_lens=len(seq)
            # 还会去读从未写入的 cache 位置 → 垃圾输出。退出后由同轮的
            # waiting 遍历或下一轮从 waiting 重新准入、重算 chunk。
            while not self.block_manager.can_append_chunk(seq, num_sched):
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim)
                    if victim is seq:
                        return False    # 自己成了牺牲品：num_sched 陈旧，勿调度
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
