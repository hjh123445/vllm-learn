# v2/engine/scheduler.py
"""v2 调度：准入判据从 slot 个数改成 block 预算（显存账本），
并支持抢占（preemption）：block 耗尽时把占块最多的请求打回 waiting 重算。"""
import torch


class Request:
    def __init__(self, req_id: int, prompt_token_ids: list[int],
                 sampling_params, eos_ids: set[int]):
        self.id = req_id
        self.prompt_ids = prompt_token_ids
        self.sp = sampling_params
        self.eos_ids = eos_ids
        self.generated = []
        self.finished = False
        self.finish_reason = None
        # 统计
        self.n_preempts = 0

    @property
    def cur_len(self) -> int:
        return len(self.prompt_ids) + len(self.generated)

    def reset(self):
        """抢占后打回重算状态。"""
        self.generated = []
        self.finished = False
        self.finish_reason = None
        self.n_preempts += 1

    def step(self, token: int):
        if self.finished:
            return
        if token in self.eos_ids and not self.sp.ignore_eos:
            self.finished = True
            self.finish_reason = "stop"
        elif len(self.generated) >= self.sp.max_tokens:
            self.finished = True
            self.finish_reason = "length"
        else:
            self.generated.append(token)


class Scheduler:
    def __init__(self, eos_ids: set[int], max_seq_len: int, max_batch_size: int, kv):
        self.eos_ids = eos_ids
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv = kv                       # KVCacheManager：准入/抢占都查它的账本
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self._next_id = 0

    def add(self, prompt_token_ids: list[int], sampling_params) -> Request:
        assert len(prompt_token_ids) > 0
        assert len(prompt_token_ids) + sampling_params.max_tokens <= self.max_seq_len
        req = Request(self._next_id, prompt_token_ids, sampling_params, self.eos_ids)
        self._next_id += 1
        self.waiting.append(req)
        return req

    def admit(self) -> list[Request]:
        """准入三条件：有等待的、并发没超 max_batch_size、block 要得起 prompt
        （外加 1 个 block 的水位储备，覆盖 decode 步的块增长）。"""
        new = []
        while (self.waiting and len(self.running) < self.max_batch_size
               and self.kv.can_allocate(len(self.waiting[0].prompt_ids), reserve_blocks=1)):
            req = self.waiting.pop(0)
            self.kv.allocate(req.id, len(req.prompt_ids))
            self.running.append(req)
            new.append(req)
        return new

    def evict(self) -> list[Request]:
        done = [r for r in self.running if r.finished]
        for r in done:
            self.running.remove(r)
            self.kv.free(r.id)
        return done

    def preempt(self, victim: Request):
        """抢占：释放全部 block，请求打回 waiting 队头，之后从头重算（recompute 策略）。"""
        self.running.remove(victim)
        self.kv.free(victim.id)
        victim.reset()
        self.waiting.insert(0, victim)

    def pick_victim(self) -> Request:
        """抢占谁：占 block 最多的（释放收益最大）。"""
        return max(self.running, key=lambda r: len(self.kv.tables[r.id]))

    @property
    def busy(self) -> bool:
        return bool(self.waiting or self.running)
