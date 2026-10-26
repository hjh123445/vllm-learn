# v2/engine/scheduler.py
"""v2 调度：显存分配和请求状态的唯一管理者。

对 engine 只暴露两个接口：
  schedule() -> (list[Request], is_prefill)   # 这一步跑什么
  update(reqs, tokens)                        # 跑完记账
准入/驱逐/抢占/block 分配全部在 scheduler 内部完成（block 预算找 KVCacheManager 要）。

调度策略：prefill 优先——有空闲 block 就先准入新请求（组 prefill 批），
prefill-priority 下在途请求的 decode 会让一步（vLLM 默认策略，简单且吞吐好）。"""
from engine.request import Request


class Scheduler:
    def __init__(self, eos_ids: set[int], max_seq_len: int, max_batch_size: int, kv):
        self.eos_ids = eos_ids
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size      # 并发数上限（计算维度；显存维度是 block 预算）
        self.kv = kv                             # KVCacheManager：唯一的显存账本
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self._next_id = 0

    # ============ 对 engine 的接口 ============

    def add(self, prompt_token_ids: list[int], sampling_params) -> Request:
        assert len(prompt_token_ids) > 0
        assert len(prompt_token_ids) + sampling_params.max_tokens <= self.max_seq_len
        req = Request(self._next_id, prompt_token_ids, sampling_params, self.eos_ids)
        self._next_id += 1
        self.waiting.append(req)
        return req

    @property
    def busy(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> tuple[list[Request], bool]:
        """返回这一步要跑的 (请求列表, 是否 prefill)。
        prefill 优先：能准入就先跑 prefill 批；否则跑全体在途请求的 decode 批。"""
        new = self._admit()
        if new:
            return new, True
        self._ensure_decode_blocks()
        return list(self.running), False

    def update(self, reqs: list[Request], tokens) -> None:
        """engine 跑完一步后记账：逐请求 step + 驱逐已完成的。"""
        for req, token in zip(reqs, tokens.tolist()):
            req.step(token)
        self.evict()

    def evict(self) -> list[Request]:
        done = [r for r in self.running if r.finished]
        for r in done:
            self.running.remove(r)
            self.kv.free(r.id)
        return done

    # ============ 内部：准入 / 抢占 ============

    def _admit(self) -> list[Request]:
        """准入三条件：有等待、并发没超 max_batch_size、block 要得起 prompt（+1 储备）。"""
        new = []
        while (self.waiting and len(self.running) < self.max_batch_size
               and self.kv.can_allocate(len(self.waiting[0].prompt_ids), reserve_blocks=1)):
            req = self.waiting.pop(0)
            self.kv.allocate(req.id, len(req.prompt_ids))
            self.running.append(req)
            new.append(req)
        return new

    def _ensure_decode_blocks(self):
        """decode 步前确保每个在途请求的写位置有 block；不够 → 抢占占块最多的（recompute）。"""
        for req in list(self.running):
            while not self.kv.ensure_append(req.id, req.cur_len):
                victim = self._pick_victim()
                self._preempt(victim)
                if victim is req:
                    break

    def _pick_victim(self) -> Request:
        return max(self.running, key=lambda r: len(self.kv.tables[r.id]))

    def _preempt(self, victim: Request):
        """抢占：释放全部 block，请求打回 waiting 队头，之后从头重算。"""
        self.running.remove(victim)
        self.kv.free(victim.id)
        victim.reset()
        self.waiting.insert(0, victim)
