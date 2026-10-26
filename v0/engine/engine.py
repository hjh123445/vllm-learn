# v0/engine/engine.py
"""v0 引擎：只做协调。
generate 主循环在这里——Batch 只管状态（get/append/is_finished），
Scheduler 只管排队（add/schedule），ModelRunner 只管计算（forward/mask/sample）。"""
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
        pad = self.tokenizer.pad_token_id or eos
        self.runner = ModelRunner(model_dir, batch_size=batch_size, max_seq_len=max_seq_len,
                                 device=device, dtype=dtype, eos=eos, **engine_kwargs)
        self.scheduler = Scheduler(eos_id=eos, pad_id=pad,
                                   max_batch_size=batch_size, max_len=max_seq_len)

    # ---------------- 对外唯一入口 ----------------

    def generate(self, prompts, sampling_params: SamplingParams | list[SamplingParams] | None = None):
        """prompts: 字符串列表或 token id 列表的列表；返回与输入顺序一致的 [{"text", "token_ids"}]"""
        sps = sampling_params or SamplingParams()
        if not isinstance(sps, list):
            sps = [sps] * len(prompts)
        token_ids = [self.tokenizer.encode(p, add_special_tokens=False)
                     if isinstance(p, str) else list(p) for p in prompts]
        self.scheduler.add(token_ids, sps)

        outs = []
        while self.scheduler.has_work:
            batch = self.scheduler.schedule()
            outs.extend(self._generate_batch(batch))
        return outs

    # ---------------- 生成主循环（engine 拥有）----------------

    @torch.no_grad()
    def _generate_batch(self, batch) -> list[dict]:
        sp = batch.params[0]      # v0 限制：整批共用第 0 条请求的 temperature/top_p
        self.runner.reset_cache()

        while not batch.is_finished():
            input_ids, positions = batch.get()
            mask = (self.runner.prefill_mask(batch) if batch.is_prefill
                    else self.runner.decode_mask(batch))
            logits = self.runner.forward(input_ids, positions, mask)
            batch.append(self.runner.sample(logits, sp.temperature, sp.top_p))

        return [{"text": self.tokenizer.decode(o["token_ids"], skip_special_tokens=True),
                 "token_ids": o["token_ids"], "finish_reason": o["finish_reason"]}
                for o in batch.collect()]
