# v7/layers/mtp.py
"""v7.2 投机解码：增量 verify（KV 轮间持久，每轮只 forward k+1 个新 token）。

v7.0 的 verify 是"整段重算"：每轮把 [prompt + 已接受 + 候选] 全部重新 prefill，
成本 O(L)，比普通 decode 还慢——性能永远无法反转。
v7.2 利用 KV cache 的本质：已确认前缀的 KV 轮间持久，每轮只 forward 新 token。

轮次结构（greedy verify）：
  状态不变量：进入每轮时，cache 含位置 0..L-2，最后一个确认 token（位置 L-1）未前向。
  1. draft：从历史找 n-gram 匹配 → k 个候选 d_0..d_{k-1}（位置 L..L+k-1）；
  2. forward [x_{L-1}, d_0..d_{k-1}]（k+1 个 token，一次 varlen 前向）：
     logits[i] = 位置 L+i 的分布 → greedy[i] 验证 d_i，greedy[k] 是全接受的 bonus；
  3. 提交：接受 d_0..d_{n_acc-1} + 修正 token greedy[n_acc]，回到不变量。

正确性关键——被拒候选的垃圾 KV：forward 会把 d_{n_acc}..d_{k-1} 的 KV 写进位置
L+n_acc+1.. 之后，这些槽位对确认序列是"未来位置"，下一轮写入前不会被读
（prefill kernel 的 key 循环界是 cu_seqlens_k=确认长度，decode 是 context_lens），
下一轮的同位置写入直接覆盖。这是 PagedAttention 的天然红利：位置 → 槽位映射
固定，垃圾不会泄漏进注意力。

成本：每轮 1 次 (k+1)-token 前向 ≈ 1 次 decode step（decode 是带宽 bound，
k+1 个 query 的增量开销小），产出期望 (1-α^{k+1})/(1-α) 个 token → 加速比。
"""

import torch

from engine.sequence import Sequence
from samplingParams import SamplingParams
from utils.context import set_context
from layers.mtp_impl import NGramDrafter      # 组件文件共存：实现拆在 mtp_impl.py


class SpeculativeEngine:
    """套在 LLMEngine 外的投机解码壳：直接驱动其 model_runner + block_manager。

    不走 scheduler（投机版自己管块表与轮次），与普通 greedy 共享同一份模型权重
    和 KV 池——切换投机/普通模式零成本，这是对比实验的前提。"""

    def __init__(self, engine, drafter: NGramDrafter):
        self.eng = engine
        self.runner = engine.model_runner
        self.bm = engine.scheduler.block_manager
        self.drafter = drafter
        self.k = drafter.k
        # 统计
        self.n_rounds = 0        # verify 轮数（含无候选时的单 token 退化为普通步）
        self.n_accepted = 0      # 被接受的 draft token 总数
        self.n_drafted = 0       # 实际产出的 draft token 总数

    @property
    def accept_rate(self) -> float:
        return self.n_accepted / self.n_drafted if self.n_drafted else 0.0

    # ---------------- 块表管理 ----------------

    def _ensure_blocks(self, seq: Sequence, total_len: int):
        """把块表扩到能容纳 total_len 个位置（投机轮要提前为候选留槽）。"""
        bs = self.bm.block_size
        need = (total_len + bs - 1) // bs
        while len(seq.block_table) < need:
            seq.block_table.append(self.bm._allocate_block())

    # ---------------- 前向 ----------------

    @torch.inference_mode()
    def _forward(self, seq: Sequence, tokens: list[int], positions: list[int]):
        """单序列 varlen 前向：把 tokens 的 KV 写进 cache，返回逐 token logits [T, V]。

        与 model_runner.prepare_prefill 同构（chunked prefill 的 q-是-k-尾段形态），
        但 token 范围由投机轮次决定，不受 scheduler 管辖。"""
        dev = self.runner.device
        bs = self.bm.block_size
        end = positions[-1] + 1
        slot_mapping = [seq.block_table[p // bs] * bs + p % bs for p in positions]
        block_tables = torch.tensor([seq.block_table], dtype=torch.int32, device=dev)
        cu_q = torch.tensor([0, len(tokens)], dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0, end], dtype=torch.int32, device=dev)
        set_context(True, cu_q, cu_k, len(tokens), end,
                    torch.tensor(slot_mapping, dtype=torch.int32, device=dev),
                    None, block_tables)
        input_ids = torch.tensor(tokens, dtype=torch.int64, device=dev)
        pos = torch.tensor(positions, dtype=torch.int64, device=dev)
        hidden = self.runner.model(input_ids, pos)          # [T, hidden]
        return self.runner.model.compute_logits(hidden)     # [T, V]

    # ---------------- 主循环 ----------------

    def generate(self, prompt_ids: list[int], max_tokens: int,
                 ignore_eos: bool = False) -> list[int]:
        """greedy 投机生成，返回 completion token ids（与 eng.generate 对齐）。"""
        eos = self.eng.config.eos
        seq = Sequence(list(prompt_ids), SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=ignore_eos))

        # ---- prefill：一次前向整个 prompt，KV 常驻 ----
        P = len(seq)
        self._ensure_blocks(seq, P)
        logits = self._forward(seq, seq.token_ids, list(range(P)))
        first = int(logits[-1].argmax())                    # 位置 P 的 greedy
        seq.append_token(first)

        # ---- 迭代 verify：不变量 = cache 含 0..L-2，最后 token 未前向 ----
        finished = (not ignore_eos and first == eos)
        while not finished and seq.num_completion_tokens < max_tokens:
            L = len(seq)                                    # 候选位置从 L 开始
            draft = self.drafter.draft(seq.token_ids)
            # forward [x_{L-1}] + draft：最后确认 token 也要前向（它的 KV 还不在 cache）
            tokens = [seq.last_token] + draft
            positions = list(range(L - 1, L - 1 + len(tokens)))
            self._ensure_blocks(seq, positions[-1] + 1)
            logits = self._forward(seq, tokens, positions)
            greedy = logits.argmax(dim=-1).tolist()         # greedy[i] = 位置 L+i

            # 接受判定：最长前缀匹配
            n_acc = 0
            for i in range(len(draft)):
                if draft[i] == greedy[i]:
                    n_acc += 1
                else:
                    break
            new_tokens = draft[:n_acc] + [greedy[n_acc]]    # 修正/bonus token

            # 统计
            self.n_rounds += 1
            self.n_drafted += len(draft)
            self.n_accepted += n_acc

            # 提交（注意：new_tokens 的 KV 中最后一个未前向——不变量保持）
            budget = max_tokens - seq.num_completion_tokens
            for t in new_tokens[:budget]:
                seq.append_token(t)
                if not ignore_eos and t == eos:
                    finished = True
                    break

        out = seq.completion_token_ids
        # 释放块（投机轮多分配的候选槽一并归还）
        self.bm.deallocate(seq)
        return out
