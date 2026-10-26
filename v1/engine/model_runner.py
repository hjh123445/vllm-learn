# v1/engine/model_runner.py
"""v1 ModelRunner（混合批次版）：一个 step_mixed 把 prefill 和 decode 在同一个 forward 里跑完。

外部形状是 flattened batch（Orca 混合批次）：
  prompt token 和 decode token 拼成一条 [total_q] 序列过模型，
  非注意力层对 total_q 无差别；attention 按 plan 分组（decode 组合批 + prefill 逐行）。
plan 由 step_mixed 一次算好、全部层共享。
"""
import torch

from config import Qwen3Config
from models.qwen3 import Qwen3ForCausalLM
from utils.loader import load_model


class MixedAttnPlan:
    """一次混合 forward 的 attention 计划（张量全部在 GPU 上）。"""

    def __init__(self, positions, row_ids, dec_idx, dec_rows, dec_pos,
                 dec_read_len, dec_mask, pf_spans, pf_masks):
        self.positions = positions          # [total_q] RoPE 位置
        self.row_ids = row_ids              # [total_q] 每个 token 的 cache 行号
        self.dec_idx = dec_idx              # [n_dec] decode token 的 flat 索引
        self.dec_rows = dec_rows            # [n_dec]
        self.dec_pos = dec_pos              # [n_dec] 被喂 token 的位置
        self.dec_read_len = dec_read_len
        self.dec_mask = dec_mask            # [n_dec, 1, 1, L]
        self.pf_spans = pf_spans            # [(i0, i1, row, L), ...]
        self.pf_masks = pf_masks            # [1, 1, s, L] 与 pf_spans 对齐
        self.n_dec = len(dec_rows)


class ModelRunner:

    def __init__(self, model_dir: str, **engine_kwargs):
        self.config = Qwen3Config.from_pretrained(model_dir, **engine_kwargs)
        cfg = self.config
        self.device, self.dtype, self.max_seq_len = cfg.device, cfg.dtype, cfg.max_seq_len
        self.model = Qwen3ForCausalLM(cfg)
        load_model(self.model, model_dir, device=cfg.device, dtype=cfg.dtype,
                   tie_word_embeddings=cfg.tie_word_embeddings)

    # ---------------- 混合批次一步：prefill + decode 同一个 forward ----------------

    @torch.no_grad()
    def step_mixed(self, new_reqs: list, running: list,
                   temperature: float, top_p: float) -> torch.Tensor:
        """new_reqs: 本轮准入（要 prefill 的）；running: 在途请求（各出 1 个 token）。
        返回 [len(new_reqs) + len(running)] 个 token，与 new_reqs + running 对齐。"""
        neg = torch.finfo(self.dtype).min
        dev = self.device
        ids, poss, rows = [], [], []
        pf_spans, pf_masks, logits_idx = [], [], []

        # ---- prefill 部分：每行 prompt 是 flat 批里一段连续切片 ----
        off = 0
        for req in new_reqs:
            L = len(req.prompt_ids)
            ids.append(torch.tensor(req.prompt_ids))
            poss.append(torch.arange(L))
            rows.append(torch.full((L,), req.slot))
            pf_spans.append((off, off + L, req.slot, L))
            causal = torch.arange(L)[:, None] >= torch.arange(L)[None, :]
            pf_masks.append(torch.where(causal, 0.0, neg).to(self.dtype).to(dev)[None, None])
            logits_idx.append(off + L - 1)                 # 每行最后一个 prompt token
            off += L
        n_pf_tok = off

        # ---- decode 部分：每行 1 个 token（flat 索引连续，天然成组）----
        for req in running:
            ids.append(torch.tensor([req.generated[-1]]))
            poss.append(torch.tensor([req.cur_len - 1]))
            rows.append(torch.tensor([req.slot]))
            logits_idx.append(off)
            off += 1

        input_ids = torch.cat(ids).to(dev)
        positions = torch.cat(poss).to(dev)
        row_ids = torch.cat(rows).to(dev)
        n_dec = len(running)

        # decode 组的 mask：slot <= 自己的位置
        if n_dec > 0:
            dec_pos = positions[n_pf_tok:]
            dec_read_len = int(dec_pos.max()) + 1
            slots = torch.arange(dec_read_len, device=dev)
            valid = slots[None, :] <= dec_pos[:, None]
            dec_mask = torch.where(valid[:, None, None], 0.0, neg).to(self.dtype)
            dec_idx = torch.arange(n_pf_tok, off, device=dev)
            dec_rows = row_ids[n_pf_tok:]
        else:
            dec_pos = dec_read_len = None
            dec_mask = dec_rows = torch.empty(0, dtype=torch.long, device=dev)
            dec_idx = torch.empty(0, dtype=torch.long, device=dev)

        plan = MixedAttnPlan(positions, row_ids, dec_idx, dec_rows, dec_pos,
                             dec_read_len, dec_mask, pf_spans, pf_masks)

        # ---- 一个 forward 过全模型（flattened），只在要出 token 的位置算 lm_head ----
        hidden = self.model(input_ids, plan)                    # [total_q, hidden]
        h_need = hidden[torch.tensor(logits_idx, device=dev)]   # [n_need, hidden]
        logits = self.model.compute_logits(h_need)
        return self.sample(logits, temperature, top_p)

    # ---------------- 采样 ----------------

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(-1)
        probs = (logits / temperature).softmax(-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = probs.sort(-1, descending=True)
            cum = sorted_probs.cumsum(-1)
            sorted_probs[cum - sorted_probs > top_p] = 0
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
            probs = probs / probs.sum(-1, keepdim=True)
        return torch.multinomial(probs, 1).squeeze(-1)
