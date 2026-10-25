# v2/engine/engine.py
"""v2 引擎：只做协调。准入按 block 预算，block 耗尽时抢占最重的请求重算。"""
import torch
from transformers import AutoTokenizer

from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler
from samplingParams import SamplingParams


class Engine:

    def __init__(self, model_dir: str, batch_size: int = 8, max_seq_len: int = 4096,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 **engine_kwargs):
        # 多余的关键字参数透传给 ModelRunner（block_size / gpu_memory_utilization 等，
        # from_pretrained 会过滤出已知字段）
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
            # 1. 准入（block 预算）+ prefill
            for req in self.scheduler.admit():
                logits = self.runner.prefill(req)
                token = self.runner.sample(logits, req.sp.temperature, req.sp.top_p)[0].item()
                req.step(token)
            self.scheduler.evict()

            # 2. decode 一步前确保每行的下一个位置有 block；不够 → 抢占占块最多的
            for req in list(self.scheduler.running):
                while not self.runner.kv.ensure_append(req.id, req.cur_len):
                    victim = self.scheduler.pick_victim()
                    if victim is req:
                        self.scheduler.preempt(victim)   # 自己就是最重的：抢自己
                        break
                    self.scheduler.preempt(victim)
                    print(f"[v2 抢占] req {victim.id} 被打回重算 "
                          f"(池剩 {self.runner.kv.num_free()} blocks)")
            if not self.scheduler.running:
                continue

            # 3. 全体活跃请求合批 decode
            running = list(self.scheduler.running)
            sp = running[0].sp
            next_tokens = self.runner.decode_step(running, sp.temperature, sp.top_p)
            for req, token in zip(running, next_tokens.tolist()):
                req.step(token)

            self.scheduler.evict()

        return [{"text": self.tokenizer.decode(r.generated, skip_special_tokens=True),
                 "token_ids": r.generated,
                 "finish_reason": r.finish_reason,
                 "n_preempts": r.n_preempts} for r in reqs]
