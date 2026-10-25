# engine/scheduler.py
"""组 batch：请求 token 进，静态 batch（左 padding、positions、生成状态）出。不做任何前向计算。"""
import torch


class Scheduler:
    def __init__(self, eos_ids: set[int], pad_token_id: int,
                 max_seq_len: int, max_batch_size: int):
        self.eos_ids = eos_ids
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

    def schedule(self, prompt_token_ids: list[list[int]],
                 sampling_params) -> "StaticBatch":
        """组一个静态 batch。prompt_token_ids: 每条请求的 token id；sampling_params: 等长 list[SamplingParams]"""
        assert len(prompt_token_ids) > 0, "batch 不能为空"
        assert len(prompt_token_ids) <= self.max_batch_size, \
            f"batch {len(prompt_token_ids)} 超过容量 {self.max_batch_size}"
        b = len(prompt_token_ids)
        assert all(len(p) > 0 for p in prompt_token_ids), "prompt 不能为空"

        prompt_lens = torch.tensor([len(p) for p in prompt_token_ids])
        max_prompt = int(prompt_lens.max())
        max_new = max(sp.max_tokens for sp in sampling_params)
        assert max_prompt + max_new <= self.max_seq_len, \
            "prompt + max_tokens 超过 cache 容量，调大 max_seq_len"

        # ---- 左 padding：所有 prompt 右对齐，最后一位都是真实 token ----
        input_ids = torch.full((b, max_prompt), self.pad_token_id, dtype=torch.long)
        for i, p in enumerate(prompt_token_ids):
            input_ids[i, max_prompt - len(p):] = torch.tensor(p)
        pad_lens = max_prompt - prompt_lens                       # [b] 每行左侧 pad 数

        # prefill 的 RoPE 位置：pad 区置 0（反正会被 mask 掉），真实区 0..L-1
        positions = torch.zeros(b, max_prompt, dtype=torch.long)
        for i, L in enumerate(prompt_lens.tolist()):
            positions[i, max_prompt - L:] = torch.arange(L)

        return StaticBatch(input_ids, prompt_lens, pad_lens, positions,
                           sampling_params, self.eos_ids, self.pad_token_id)


class StaticBatch:
    """一次静态批处理的全部状态：输入张量 + 每条请求的生成进度。"""

    def __init__(self, input_ids: torch.Tensor, prompt_lens: torch.Tensor,
                 pad_lens: torch.Tensor, positions: torch.Tensor,
                 sampling_params, eos_ids: set[int], pad_token_id: int):
        self.input_ids = input_ids          # [b, s] 左 padding 的 prompt
        self.prompt_lens = prompt_lens      # [b]
        self.pad_lens = pad_lens            # [b] 左侧 pad 数（attention mask 用）
        self.positions = positions          # [b, s] prefill 的 RoPE 位置
        self.params = sampling_params       # list[SamplingParams]
        self.eos_ids = eos_ids
        self.pad_token_id = pad_token_id

        self.generated = [[] for _ in range(input_ids.shape[0])]
        self.finished = [False] * input_ids.shape[0]

    # ================= 查询 =================

    @property
    def bsz(self) -> int:
        return self.input_ids.shape[0]

    @property
    def max_prompt_len(self) -> int:
        return self.input_ids.shape[1]

    @property
    def all_finished(self) -> bool:
        return all(self.finished)

    def decode_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        """下一步要喂的 ([b,1] token, [b,1] 位置)。位置 = prompt_len + 已生成数 - 1；
        刚 prefill 完就 eos 的行没有生成 token，喂 pad 维持 batch 对齐。"""
        toks, poss = [], []
        for i in range(self.bsz):
            if self.generated[i]:
                toks.append(self.generated[i][-1])
                poss.append(self.prompt_lens[i].item() + len(self.generated[i]) - 1)
            else:
                toks.append(self.pad_token_id)
                poss.append(self.prompt_lens[i].item() - 1)
        return (torch.tensor(toks)[:, None], torch.tensor(poss)[:, None])

    # ================= 推进 =================

    def step(self, next_tokens: torch.Tensor):
        """采样出的 [b] 个 token 记账；eos / 达到 max_tokens 即完成该请求。"""
        for i, t in enumerate(next_tokens.tolist()):
            if self.finished[i]:
                continue
            sp = self.params[i]
            if (t in self.eos_ids and not sp.ignore_eos) \
                    or len(self.generated[i]) >= sp.max_tokens:
                self.finished[i] = True
            else:
                self.generated[i].append(t)

    # ================= 输出 =================

    def collect(self) -> list[list[int]]:
        return [list(g) for g in self.generated]
