# v2/engine/request.py
"""单个请求的全部状态。scheduler 管理它的流转，model 只见 token。"""


class Request:
    def __init__(self, req_id: int, prompt_token_ids: list[int],
                 sampling_params, eos_ids: set[int]):
        self.id = req_id
        self.prompt_ids = prompt_token_ids
        self.sp = sampling_params
        self.eos_ids = eos_ids
        self.generated = []          # 已生成的 token（不含 prompt）
        self.finished = False
        self.finish_reason = None    # "stop" | "length"
        self.n_preempts = 0

    @property
    def cur_len(self) -> int:
        """prompt + 已生成 = 下一个 token 的位置（也是 cache 里已占的长度）。"""
        return len(self.prompt_ids) + len(self.generated)

    def reset(self):
        """被抢占后打回重算状态。"""
        self.generated = []
        self.finished = False
        self.finish_reason = None
        self.n_preempts += 1

    def step(self, token: int):
        """记账一个采样出的 token；eos / 达到 max_tokens 即完成。"""
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
