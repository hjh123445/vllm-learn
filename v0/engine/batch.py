# v0/engine/batch.py
"""一个静态批处理的全部状态。

序列缓冲区一次性预分配 [b, max_prompt_len + max_tokens]：左边放 prompt（左 padding），
右边逐步写入生成的 token。prefill/decode 都从这块 buffer 取视图，positions 全长预计算。
Batch 只管状态（get/append/collect），不做任何计算。
"""
import torch


class Batch:
    def __init__(self, prompt_ids: list[list[int]], sampling_params,
                 eos_id: int, pad_id: int, max_len: int):
        assert len(prompt_ids) > 0 and all(len(p) > 0 for p in prompt_ids)
        self.eos_id, self.pad_id = eos_id, pad_id
        self.params = sampling_params                    # list[SamplingParams]，与行对齐
        self.prompt_len = [len(p) for p in prompt_ids]
        self.max_prompt_len = max(self.prompt_len)
        self.bsz = len(prompt_ids)
        self.max_tokens = max(sp.max_tokens for sp in sampling_params)
        assert self.max_prompt_len + self.max_tokens <= max_len, "超过 cache 容量"

        W = self.max_prompt_len + self.max_tokens        # 序列缓冲区全长
        # ---- 左 padding 的 prompt 区 + 逐步填充的生成区 ----
        self.seq = torch.full((self.bsz, W), pad_id, dtype=torch.long)
        for i, p in enumerate(prompt_ids):
            self.seq[i, self.max_prompt_len - len(p):self.max_prompt_len] = torch.tensor(p)
        # ---- 位置一次算全长：pad 区 clamp 0（会被 mask 掉），生成区自动接续 ----
        # 第 i 行第 j 列: j - pad_lens[i]（prompt 区 0..L-1，生成区 L, L+1, ...）
        self.pad_lens = torch.tensor([self.max_prompt_len - L for L in self.prompt_len])
        ar = torch.arange(W)
        self.positions = (ar[None, :] - self.pad_lens[:, None]).clamp(min=0)   # [b, W]

        # ---- 生成状态 ----
        self.is_prefill = True
        self.gen_len = 0                                 # 已生成列数（整批同步推进）
        self.finished = [False] * self.bsz
        self.finish_reason = [None] * self.bsz

    # ============ 引擎每步取输入 ============

    @property
    def cache_len(self) -> int:
        """cache 里已写入的长度（含 prompt 的左 pad）。
        gen_len 在 append 时就 +1，但生成的 token 要下一步"喂进去"才写 cache，
        所以 cache 长度 = max_prompt_len + gen_len - 1。"""
        return self.max_prompt_len + max(self.gen_len - 1, 0)

    def cur_len(self) -> int:   # 兼容旧名
        return self.cache_len

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回这一步要喂的 (input_ids, positions)。
        prefill: 整个 prompt 区；decode: 上一步生成 token 所在的那一列。"""
        if self.is_prefill:
            return self.seq[:, :self.max_prompt_len], self.positions[:, :self.max_prompt_len]
        col = self.max_prompt_len + self.gen_len - 1
        return self.seq[:, col:col + 1], self.positions[:, col:col + 1]

    # ============ 引擎每步记输出 ============

    def append(self, next_tokens: torch.Tensor):
        """把采样出的 [b] 个 token 写进生成区下一列；eos / 达到 max_tokens 即完成。"""
        col = self.max_prompt_len + self.gen_len
        self.seq[:, col] = next_tokens.cpu()
        self.gen_len += 1
        self.is_prefill = False
        for i, t in enumerate(next_tokens.tolist()):
            sp = self.params[i]
            if t == self.eos_id and not sp.ignore_eos:
                self.finished[i] = True
                self.finish_reason[i] = "stop"
            elif self.gen_len >= sp.max_tokens:
                self.finished[i] = True
                self.finish_reason[i] = "length"

    def is_finished(self) -> bool:
        return all(self.finished)

    # ============ 输出 ============

    def collect(self) -> list[dict]:
        """每行取生成区里自己的 token：eos 截断 / max_tokens 截断。"""
        outs = []
        for i in range(self.bsz):
            row = self.seq[i, self.max_prompt_len:self.max_prompt_len + self.gen_len].tolist()
            if self.finish_reason[i] == "stop" and self.eos_id in row:
                row = row[:row.index(self.eos_id)]
            outs.append({"token_ids": row, "finish_reason": self.finish_reason[i]})
        return outs
