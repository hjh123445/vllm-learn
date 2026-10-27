# v2/engine/llm_engine.py
"""v2 引擎：只做协调，主循环只有 7 行。

schedule() 给什么就跑什么：scheduler 管显存分配/准入/抢占/驱逐，
runner 只做平铺前向 + 采样，engine 把两边接起来。"""
import torch
from transformers import AutoTokenizer

from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler
from samplingParams import SamplingParams


class Engine:

    def __init__(self, model_dir: str, batch_size: int = 8, max_seq_len: int = 4096,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 **engine_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        eos = self.tokenizer.eos_token_id
        self.runner = ModelRunner(model_dir, batch_size=batch_size, max_seq_len=max_seq_len,
                                  device=device, dtype=dtype, eos=eos, **engine_kwargs)
        self.scheduler = Scheduler(eos_ids={eos}, max_seq_len=max_seq_len,
                                   max_batch_size=batch_size, kv=self.runner.kv)

    # ---------------- 对外唯一入口 ----------------

    def generate(self, prompts, sampling_params: SamplingParams | list[SamplingParams] | None = None):
        sps = sampling_params or SamplingParams()
        if not isinstance(sps, list):
            sps = [sps] * len(prompts)
        assert len(sps) == len(prompts)

        token_ids = [self.tokenizer.encode(p, add_special_tokens=False)
                     if isinstance(p, str) else list(p) for p in prompts]
        reqs = [self.scheduler.add(ids, sp) for ids, sp in zip(token_ids, sps)]

        while self.scheduler.busy:
            batch, is_prefill = self.scheduler.schedule()
            if not batch:
                continue
            sp = batch[0].sp          # v2 限制：整批共用第一个请求的采样参数
            tokens = self.runner.step(batch, is_prefill, sp.temperature, sp.top_p)
            self.scheduler.update(batch, tokens)

        return [{"text": self.tokenizer.decode(r.generated, skip_special_tokens=True),
                 "token_ids": r.generated,
                 "finish_reason": r.finish_reason,
                 "n_preempts": r.n_preempts} for r in reqs]
