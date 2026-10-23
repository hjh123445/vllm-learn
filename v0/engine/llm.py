# engine/llm.py
"""引擎最外层：文本↔token 只在这层发生，生成循环、padding、mask 也在这层。"""
import torch
from transformers import AutoTokenizer

from engine.model_runner import ModelRunner
from samplingParams import SamplingParams


class LLM:

    def __init__(self, model_dir: str, batch_size: int = 8, max_seq_len: int = 4096,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 **ignored):                       # 吃掉 enforce_eager/tensor_parallel_size 等
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.eos_id = self.tokenizer.eos_token_id
        self.batch_size, self.max_seq_len = batch_size, max_seq_len
        self.device, self.dtype = device, dtype
        self.runner = ModelRunner(model_dir, batch_size=batch_size, max_seq_len=max_seq_len,
                                  device=device, dtype=dtype, eos=self.eos_id)

    # ---------------- 对外唯一入口 ----------------

    @torch.no_grad()
    def generate(self, prompts, sampling_params: SamplingParams | list[SamplingParams] | None = None):
        """
        prompts: 字符串列表（调用方自己套 chat_template，参考 nano-vllm 的 example）
                 或已经 tokenize 好的 token id 列表的列表
        返回:     [{"text": ..., "token_ids": ...}, ...] 与输入顺序一致
        """
        sps = sampling_params or SamplingParams()
        if not isinstance(sps, list):
            sps = [sps] * len(prompts)
        max_new = max(sp.max_tokens for sp in sps)

        # ---- 1. tokenize（引擎职责，模型永远只见 token id）----
        token_ids = [self.tokenizer.encode(p, add_special_tokens=False)
                     if isinstance(p, str) else list(p) for p in prompts]
        prompt_lens = [len(t) for t in token_ids]
        bsz, max_prompt = len(prompts), max(prompt_lens)
        assert bsz <= self.batch_size, f"batch {bsz} 超过容量 {self.batch_size}"
        assert max_prompt + max_new <= self.max_seq_len, "超过 cache 容量，调大 max_seq_len"

        # ---- 2. 组静态 batch：左 padding，末尾对齐 ----
        pad_id = self.tokenizer.pad_token_id or self.eos_id
        input_ids = torch.full((bsz, max_prompt), pad_id, dtype=torch.long)
        for i, t in enumerate(token_ids):
            input_ids[i, max_prompt - len(t):] = torch.tensor(t)
        pad_lens = torch.tensor([max_prompt - L for L in prompt_lens])   # [b] 左pad宽度

        # ---- 3. 位置与 mask（真实位置坐标系：prompt 占 0..L-1）----
        neg = torch.finfo(self.dtype).min

        def prefill_positions() -> torch.Tensor:
            pos = torch.zeros(bsz, max_prompt, dtype=torch.long)
            for i, L in enumerate(prompt_lens):
                pos[i, max_prompt - L:] = torch.arange(L)
            return pos

        def build_mask(query_slots, cur_len: int) -> torch.Tensor:
            """causal ∧ 非pad，[b,1,s_q,cur_len]，可见0 不可见-inf
            query_slots: 各 query token 在 cache 里的 slot 号（prefill=0..s-1，decode=cur_len-1）"""
            slots = torch.arange(cur_len)
            valid = slots[None, :] >= pad_lens[:, None]                    # [b,cur_len]
            qslots = torch.as_tensor(query_slots)
            causal = qslots[:, None] >= slots[None, :]                     # [s_q,cur_len]
            allowed = causal[None, None] & valid[:, None, None]            # [b,1,s_q,cur_len]
            return torch.where(allowed, 0.0, neg).to(self.dtype)

        # ---- 4. prefill → decode 循环 ----
        self.runner.reset_cache()                                          # 新batch：重置一次
        outputs, finished = [[] for _ in range(bsz)], [False] * bsz
        prev_tok, next_tok = None, None

        for step in range(max_new):
            if step == 0:
                logits = self.runner.forward(input_ids, prefill_positions(),
                                             build_mask(list(range(max_prompt)), max_prompt))
            else:
                pos = torch.tensor([[L + step - 1] for L in prompt_lens])  # 每请求真实位置
                # cache 即将写入 cur_len 这个 slot，写入后共 cur_len+1 个 slot，
                # 新 token 是最后一个，可以看到之前所有 slot
                logits = self.runner.forward(prev_tok, pos,
                                             build_mask([self.runner.cur_len],
                                                        self.runner.cur_len + 1))
            next_tok = self.runner.sample(logits, sps[0].temperature, sps[0].top_p)

            for i, t in enumerate(next_tok.tolist()):
                if finished[i]:
                    continue
                if (t == self.eos_id and not sps[i].ignore_eos) \
                        or len(outputs[i]) >= sps[i].max_tokens:
                    finished[i] = True
                else:
                    outputs[i].append(t)
            if all(finished):
                break
            prev_tok = next_tok.unsqueeze(1)     # 含finished请求，保持batch对齐

        # ---- 5. detokenize ----
        return [{"text": self.tokenizer.decode(o, skip_special_tokens=True),
                 "token_ids": o} for o in outputs]