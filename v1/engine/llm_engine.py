# v1/engine/llm_engine.py
"""v1 引擎（混合批次版）：每个迭代只有一个 forward——
新准入的 prefill 和在途的 decode 在同一个 flattened batch 里跑（Orca 混合批次）。"""
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
                                   max_batch_size=batch_size)

    # ---------------- 对外唯一入口 ----------------

    def generate(self, prompts, sampling_params: SamplingParams | list[SamplingParams] | None = None):
        sps = sampling_params or SamplingParams()
        if not isinstance(sps, list):
            sps = [sps] * len(prompts)
        assert len(sps) == len(prompts)

        token_ids = [self.tokenizer.encode(p, add_special_tokens=False)
                     if isinstance(p, str) else list(p) for p in prompts]
        reqs = [self.scheduler.add(ids, sp) for ids, sp in zip(token_ids, sps)]

        # 混合批次主循环：每个迭代 = 准入 → 一次 forward（新请求 prefill + 在途 decode 混跑）→ 驱逐
        while self.scheduler.busy:
            new = self.scheduler.admit()
            new_ids = {r.id for r in new}
            running = [r for r in self.scheduler.running if r.id not in new_ids]
            if not (new or running):
                continue                     # 全在 waiting（防御性兜底）
            sp = (new + running)[0].sp       # v1 限制：整批共用第一个请求的采样参数
            next_tokens = self.runner.step_mixed(new, running, sp.temperature, sp.top_p)
            for req, token in zip(new + running, next_tokens.tolist()):
                req.step(token)
            self.scheduler.evict()

        return [{"text": self.tokenizer.decode(r.generated, skip_special_tokens=True),
                 "token_ids": r.generated,
                 "finish_reason": r.finish_reason} for r in reqs]
