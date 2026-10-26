# v1/engine/engine.py
"""v1 引擎：只做协调。迭代级主循环——每个迭代之间准入新请求、驱逐完成的请求。"""
import torch
from transformers import AutoTokenizer

from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler
from samplingParams import SamplingParams


class Engine:

    def __init__(self, model_dir: str, batch_size: int = 8, max_seq_len: int = 4096,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 **ignored):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        eos = self.tokenizer.eos_token_id
        self.runner = ModelRunner(model_dir, batch_size=batch_size, max_seq_len=max_seq_len,
                                  device=device, dtype=dtype, eos=eos)
        self.scheduler = Scheduler(eos_ids={eos}, max_seq_len=max_seq_len,
                                   max_batch_size=batch_size)

    # ---------------- 对外唯一入口 ----------------

    def generate(self, prompts, sampling_params: SamplingParams | list[SamplingParams] | None = None):
        """
        prompts: 字符串列表（调用方自己套 chat_template）或 token id 列表的列表
        返回:     [{"text": ..., "token_ids": ...}, ...] 与输入顺序一致
        """
        sps = sampling_params or SamplingParams()
        if not isinstance(sps, list):
            sps = [sps] * len(prompts)
        assert len(sps) == len(prompts), "SamplingParams 数量要和 prompts 对齐"

        token_ids = [self.tokenizer.encode(p, add_special_tokens=False)
                     if isinstance(p, str) else list(p) for p in prompts]
        reqs = [self.scheduler.add(ids, sp) for ids, sp in zip(token_ids, sps)]

        # 迭代级主循环：每个迭代 = 准入 → 合批 prefill → 全体活跃请求合批 decode → 驱逐
        while self.scheduler.busy:
            # 1. 准入 + 合批 prefill（新请求一次算完，不再逐条）
            new = self.scheduler.admit()
            if new:
                sp = new[0].sp
                next_tokens = self.runner.prefill_batch(new, sp.temperature, sp.top_p)
                for req, token in zip(new, next_tokens.tolist()):
                    req.step(token)
            self.scheduler.evict()
            if not self.scheduler.running:
                continue

            # 2. 全体 running 请求合批 decode 一步
            reqs_running = list(self.scheduler.running)
            sp = reqs_running[0].sp   # v0/v1 共同的限制：整批共用第一个请求的采样参数
            next_tokens = self.runner.decode_step(reqs_running, sp.temperature, sp.top_p)
            for req, token in zip(reqs_running, next_tokens.tolist()):
                req.step(token)

            # 3. 驱逐完成的请求（slot 立即回收，下一迭代可准入新请求）
            self.scheduler.evict()

        return [{"text": self.tokenizer.decode(r.generated, skip_special_tokens=True),
                 "token_ids": r.generated,
                 "finish_reason": r.finish_reason} for r in reqs]
