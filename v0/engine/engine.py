# engine/engine.py
"""引擎：只做协调 —— tokenize → scheduler 组 batch → runner 跑生成 → detokenize。"""
import torch
from transformers import AutoTokenizer

from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler
from samplingParams import SamplingParams


class Engine:

    def __init__(self, model_dir: str, batch_size: int = 8, max_seq_len: int = 4096,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 **ignored):                       # 吃掉 enforce_eager/tensor_parallel_size 等
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id or eos
        self.runner = ModelRunner(model_dir, batch_size=batch_size, max_seq_len=max_seq_len,
                                  device=device, dtype=dtype, eos=eos)
        self.scheduler = Scheduler(eos_ids={eos}, pad_token_id=pad,
                                    max_seq_len=max_seq_len, max_batch_size=batch_size)

    # ---------------- 对外唯一入口 ----------------

    def generate(self, prompts, sampling_params: SamplingParams | list[SamplingParams] | None = None):
        """
        prompts: 字符串列表（调用方自己套 chat_template）
                 或已经 tokenize 好的 token id 列表的列表
        返回:     [{"text": ..., "token_ids": ...}, ...] 与输入顺序一致
        """
        sps = sampling_params or SamplingParams()
        if not isinstance(sps, list):
            sps = [sps] * len(prompts)
        assert len(sps) == len(prompts), "SamplingParams 数量要和 prompts 对齐"

        # 1. tokenize（引擎职责，模型永远只见 token id）
        token_ids = [self.tokenizer.encode(p, add_special_tokens=False)
                     if isinstance(p, str) else list(p) for p in prompts]

        # 2. 组静态 batch
        batch = self.scheduler.schedule(token_ids, sps)

        # 3. prefill + decode 主循环
        self.runner.run(batch)

        # 4. detokenize
        return [{"text": self.tokenizer.decode(o, skip_special_tokens=True),
                 "token_ids": o} for o in batch.collect()]
