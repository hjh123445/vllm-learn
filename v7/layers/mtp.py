# v7/layers/mtp.py
"""v7 MTP 投机解码：draft-verify 循环。

算法（DeepSeek-V3 的 MTP 推理用法 / 标准 speculative decoding）:
  1. draft: 用轻量 draft 模型（n-gram 或 MTP 层）自回归生成 k 个候选 token
  2. verify: 把 k 个候选拼成序列，**一次前向**算出每个位置的概率
     p_target(t_1..t_k | prefix, t_1..t_{i-1})
  3. 接受: 从左到右逐个检查——target 的 argmax == draft 的候选则接受，
     遇到第一个分歧则用 target 的 token 替换并停止
  4. 循环: 接受了 j 个 token（j<=k），产出 j+1 个 token（含分歧处 target 的）

加速比 = 平均接受长度 α（每轮产出 α+1 token 花 1 次 verify + k 次 draft）:
  加速 ≈ (α+1) / (1 + k·c_draft/c_target)，draft 越轻、α 越高越快。

本实现的 draft 用 **n-gram 匹配**（零模型成本的热身版）：
  在 prompt + 已生成里找最近的 n-gram 前缀，取其后续 k 个 token 当候选。
  后续接 MTP 层（Qwen3-30B-A3B 的 MTP 权重）。
verify 复用主引擎的 prefill 路径（平铺批的天然优势：k 个候选 token
就是 k 个"位置"，一次 forward 出全部 logits——v1 平铺批设计的直接回报）。
"""
import torch


class NGramDrafter:
    """n-gram draft：在上下文里找与当前尾部匹配的 n-gram，取其后续当候选。"""

    def __init__(self, n: int = 2, k: int = 4):
        self.n, self.k = n, k

    def draft(self, tokens: list[int]) -> list[int]:
        """tokens: prompt + 已生成的完整序列 → k 个候选 token。"""
        if len(tokens) < self.n + 1:
            return []
        # 从后往前找最近的 [t_{-n}..t_{-1}] 匹配
        tail = tuple(tokens[-self.n:])
        best = None
        for i in range(len(tokens) - self.n - 1, -1, -1):
            if tuple(tokens[i:i + self.n]) == tail:
                best = i
                break
        if best is None:
            return []
        cand = tokens[best + self.n:best + self.n + self.k]
        return cand[:self.k]


class SpeculativeEngine:
    """包一个 v6 引擎跑 draft-verify 循环。"""

    def __init__(self, engine, drafter=None):
        self.engine = engine          # v6 的 LLMEngine
        self.drafter = drafter or NGramDrafter(n=2, k=4)
        # 统计
        self.n_rounds = 0
        self.n_accepted = 0
        self.n_generated = 0

    @property
    def accept_rate(self) -> float:
        return self.n_accepted / max(self.n_rounds * self.drafter.k, 1)

    def generate(self, prompt_ids: list[int], max_tokens: int = 64,
                 temperature: float = 0.0):
        """单请求投机生成（简化版：draft-verify 串行，无 batch）。"""
        from samplingParams import SamplingParams
        seq = None
        # 用引擎的底层组件手动驱动（绕过 scheduler 的多请求逻辑——投机版专用）
        eng = self.engine
        sch = eng.scheduler
        runner = eng.model_runner
        from engine.sequence import Sequence
        sp = type("SP", (), {"temperature": temperature, "max_tokens": max_tokens,
                             "ignore_eos": True})()
        seq = Sequence(prompt_ids, sp)
        sch.add(seq)

        tokens = list(prompt_ids)
        n_out = 0
        while n_out < max_tokens and not seq.is_finished:
            # ---- 1. draft: k 个候选 ----
            cand = self.drafter.draft(tokens)
            cand = cand[:max(0, max_tokens - n_out)]
            k = len(cand)
            if k == 0:
                # draft 无候选 → 普通一步: 全量前向取末位 argmax
                logits = self._verify_forward(tokens, 0)
                t = int(logits[-1].argmax())
                seq.append_token(t)
                n_out += 1
                self.n_rounds += 1
                self.n_generated += 1
                tokens = prompt_ids + seq.completion_token_ids
                continue

            # ---- 2. verify: 全量 prefill 一次算 logits ----
            # verify_ids = 当前序列 + 候选；从每个候选前一位置的分布验证它
            verify_ids = tokens + cand
            logits = self._verify_forward(verify_ids, len(cand))
            # logits: [k, vocab] —— 每个候选位置的 target 分布

            # ---- 3. 接受判定（greedy 版）----
            greedy = logits.argmax(-1).tolist()     # target 的选择
            n_acc = 0
            for i, c in enumerate(cand):
                if greedy[i] == c:
                    n_acc += 1
                else:
                    break
            # 接受 n_acc 个候选 + 分歧处（或全部接受后）的 target token
            next_tok = greedy[n_acc] if n_acc < k else None
            # 若全部接受，用最后位置的下一个分布——logits 只有 k 个位置，
            # 全接受时 next_tok 需要 extra forward；简化: 下轮 draft 前
            # 用 greedy[k-1] 位置的 logits 已含——重算下面补
            accepted = cand[:n_acc]
            for t in accepted:
                if t != 0 or k > 1:           # 跳过占位候选
                    seq.append_token(t)
                    n_out += 1
            if next_tok is not None:
                seq.append_token(next_tok)
                n_out += 1
            elif n_acc == 0:
                # 零接受且无分歧点（k=0 退化路径已处理）——防御性跳出
                break
            # 记账
            self.n_rounds += 1
            self.n_accepted += n_acc
            self.n_generated += n_acc + (1 if next_tok is not None else 0)
            if next_tok is None:
                # 全部接受（k 个候选都对）——产出 k 个，下一步继续
                pass
            tokens = prompt_ids + seq.completion_token_ids

        out = list(seq.completion_token_ids)
        # 清理 scheduler 状态：seq 可能滞留在 waiting（手动路径不走 schedule）
        # 或 running；block 必须 deallocate 否则 block_manager 记账错
        if seq in sch.running:
            sch.running.remove(seq)
        if seq in sch.waiting:
            sch.waiting.remove(seq)
        if seq.block_table:
            sch.block_manager.deallocate(seq)
        return out

    def _verify_forward(self, verify_ids: list[int], k: int) -> torch.Tensor:
        """把 verify_ids 的前缀当已缓存，最后 k+1 个位置一次前向。
        简化实现：全量 prefill（不利用前缀 cache——正确性优先，
        性能优化在 7.2 接 APC 前缀缓存）。"""
        runner = self.engine.model_runner
        from engine.sequence import Sequence
        from types import SimpleNamespace
        # 直接调 runner.prefill 路径: 构造一个假 seq
        # prompt = verify_ids[:k+1]? 不——需要全部位置的 logits
        # 用 schedule+run 的 prefill 分支
        import torch
        L = len(verify_ids)
        # 手动构造 prefill 输入
        input_ids = torch.tensor([verify_ids])       # [1, L]
        positions = torch.arange(L)[None]
        # 需要分配 block
        # 借用 runner 现有接口最干净的方式：临时 seq
        sp = SimpleNamespace(temperature=0.0, max_tokens=1, ignore_eos=True)
        tmp = Sequence(verify_ids, sp)
        from utils.context import get_context
        # allocate blocks
        runner_config = runner.config
        nb = runner.block_size_ if hasattr(runner, "block_size_") else 256
        # runner.kv 相关——用 engine.scheduler 的 block_manager
        bm = self.engine.scheduler.block_manager
        need = bm.can_allocate(tmp)
        bm.allocate(tmp, max(need, 0))
        # context 构造（照抄 prepare_prefill 的核心）
        from utils.context import set_context
        bs = bm.block_size
        cu_q = [0, L]
        cu_k = [0, L]
        slot = []
        for pos in range(L):
            slot.append(tmp.block_table[pos // bs] * bs + pos % bs)
        import torch as _t
        bt = _t.tensor([tmp.block_table[: (L + bs - 1) // bs]], dtype=_t.int32)
        dev = "cuda"
        set_context(True, _t.tensor(cu_q, dtype=_t.int32, device=dev),
                    _t.tensor(cu_k, dtype=_t.int32, device=dev), L, L,
                    _t.tensor(slot, dtype=_t.int32, device=dev), None,
                    bt.to(dev))
        logits = runner.run_model_full(input_ids, positions) if hasattr(
            runner, "run_model_full") else None
        # runner 没有直接暴露——用 model 直调
        hidden = runner.model(input_ids.cuda().squeeze(0),
                              positions.cuda().squeeze(0))
        logits = runner.model.lm_head(hidden)        # [L, vocab]
        # 取最后 k 个位置的 logits（候选位置的 target 分布）
        bm.deallocate(tmp)
        # logits[j] 预测位置 j+1 的 token。候选 cand[i] 在位置 L-k+i，
        # 其 target 分布是 logits[L-k+i-1] → 取 [L-k-1 : L-1]。
        # k=0 时返回 logits[-1:]（预测下一个新 token——退化路径用）
        if k == 0:
            return logits[-1:].cpu()
        # 清理 verify 的 cache 痕迹（手动路径不走 run() 的管理——
        # 下一请求的 prepare 假设 cache 从 cur_len=0 开始）
        for m in runner.model.modules():
            if hasattr(m, "cache") and hasattr(m.cache, "cur_len"):
                m.cache.cur_len = 0
        return logits[len(verify_ids) - k - 1: -1].cpu()
