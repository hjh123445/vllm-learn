# v1/engine/scheduler.py
"""迭代级调度：请求在 waiting/running 两个队列间流转，
每个迭代之间可以准入新请求、驱逐已完成的请求——这就是和 v0 静态批处理的本质区别。"""
import torch


class Request:
    def __init__(self, req_id: int, prompt_token_ids: list[int],
                 sampling_params, eos_ids: set[int]):
        self.id = req_id
        self.prompt_ids = prompt_token_ids
        self.sp = sampling_params
        self.eos_ids = eos_ids
        self.slot = None          # cache 行号，准入时分配
        self.generated = []       # 已生成的 token（不含 prompt）
        self.finished = False
        self.finish_reason = None  # "stop" | "length"

    @property
    def cur_len(self) -> int:
        """prompt + 已生成 = 该请求在 cache 里占的长度（也是下一个 token 的位置）"""
        return len(self.prompt_ids) + len(self.generated)

    def step(self, token: int):
        """记账一个采样出的 token；eos / 达到 max_tokens 即完成"""
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
    def __init__(self, eos_ids: set[int], max_seq_len: int, max_batch_size: int):
        self.eos_ids = eos_ids
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.free_slots = list(range(max_batch_size))
        self._next_id = 0

    def add(self, prompt_token_ids: list[int], sampling_params) -> Request:
        assert len(prompt_token_ids) > 0, "prompt 不能为空"
        assert len(prompt_token_ids) + sampling_params.max_tokens <= self.max_seq_len, \
            "prompt + max_tokens 超过 cache 容量"
        req = Request(self._next_id, prompt_token_ids, sampling_params, self.eos_ids)
        self._next_id += 1
        self.waiting.append(req)
        return req

    def admit(self) -> list[Request]:
        """有空闲 slot 就从 waiting 准入（返回新准入的，调用方要先 prefill 它们）"""
        new = []
        while self.waiting and self.free_slots:
            req = self.waiting.pop(0)
            req.slot = self.free_slots.pop(0)
            self.running.append(req)
            new.append(req)
        return new

    def evict(self) -> list[Request]:
        """驱逐已完成的请求，slot 回收可复用"""
        done = [r for r in self.running if r.finished]
        for r in done:
            self.running.remove(r)
            self.free_slots.append(r.slot)
        return done

    @property
    def busy(self) -> bool:
        return bool(self.waiting or self.running)
