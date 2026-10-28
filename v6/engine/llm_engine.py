# v5/engine/llm_engine.py
"""v3 LLMEngine（重写版，单卡单进程，无 mp/共享内存/分布式）。

引擎只做协调：tokenizer → Scheduler（nano 的，原封不动）→ ModelRunner（我们的）。
scheduler 管准入/驱逐/抢占/chunked prefill/APC，ModelRunner 管前向和采样。"""
from dataclasses import fields

import torch
from transformers import AutoTokenizer

from config import Config
from engine.sequence import Sequence
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model: str, dtype=None, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        Sequence.block_size = self.config.kvcache_block_size
        self.model_runner = ModelRunner(self.config, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        eos = self.tokenizer.eos_token_id
        self.config.eos = eos[0] if isinstance(eos, list) else eos
        self.scheduler = Scheduler(self.config)

    # ---------------- 对外接口（与 nano 的 LLMEngine 一致）----------------

    def add_request(self, prompt: str | list[int], sampling_params) -> Sequence:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.run(seqs, is_prefill)
        # 多卡(EP/TP)时: 各卡的 argmax 可能在概率并列处翻转 → eos 判定分叉 →
        # scheduler 决策分叉 → 集合通信错位死锁。rank0 广播采样结果消除分叉。
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            t = torch.tensor(token_ids, device="cuda")
            dist.broadcast(t, src=0)
            token_ids = t.tolist()
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

    # ---------------- 便捷入口（我们自己的 generate）----------------

    def generate(self, prompts, sampling_params=None):
        """prompts: 字符串列表或 token id 列表的列表；返回与输入顺序一致的
        [{"text", "token_ids", "finish_reason"}]。"""
        if sampling_params is None:
            from samplingParams import SamplingParams
            sampling_params = SamplingParams(temperature=1e-6)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        assert len(sampling_params) == len(prompts)

        seqs = []
        for p, sp in zip(prompts, sampling_params):
            seqs.append(self.add_request(p, sp))
        while not self.is_finished():
            self.step()

        outs = []
        for seq in seqs:
            toks = seq.completion_token_ids
            eos = self.config.eos
            finish = "stop" if (not seq.ignore_eos and eos in toks) else "length"
            outs.append({"text": self.tokenizer.decode(toks, skip_special_tokens=True),
                         "token_ids": list(toks), "finish_reason": finish})
        return outs
