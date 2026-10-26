# v0/engine/scheduler.py
"""静态批调度：请求进 waiting 队列，schedule() 按 max_batch_size 组 Batch 出。
v0 是静态批处理：Batch 一组好就跑到整批完成，slot 不回收（这正是气泡的来源，v1 修掉）。"""
from collections import deque

from engine.batch import Batch


class Scheduler:
    def __init__(self, eos_id: int, pad_id: int, max_batch_size: int, max_len: int):
        self.eos_id, self.pad_id = eos_id, pad_id
        self.max_batch_size, self.max_len = max_batch_size, max_len
        self.waiting: deque = deque()      # [(prompt_ids, sp), ...]

    def add(self, prompts_ids, sampling_params) -> None:
        """入队。可以和生成循环并发调用（deque 的 append/popleft 在 CPython 里原子）。"""
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts_ids)
        assert len(sampling_params) == len(prompts_ids)
        self.waiting.extend(zip(prompts_ids, sampling_params))

    @property
    def has_work(self) -> bool:
        return bool(self.waiting)

    def schedule(self) -> "Batch | None":
        """组一个静态 batch。不够 max_batch_size 也照组（剩下留给下一个 batch）。"""
        take = min(len(self.waiting), self.max_batch_size)
        items = [self.waiting.popleft() for _ in range(take)]
        prompt_ids = [p for p, _ in items]
        sps = [sp for _, sp in items]
        return Batch(prompt_ids, sps, eos_id=self.eos_id, pad_id=self.pad_id,
                     max_len=self.max_len)
