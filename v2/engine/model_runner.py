# v2/engine/model_runner.py
"""v2 ModelRunner：显存的唯一所有者 + 平铺前向。

step(reqs, is_prefill) 是唯一的前向入口：
  1. 把这一步的请求摊平成 [total_q, hidden]；
  2. 算好 slot 映射 / decode 组 / prefill 行 → set_context 一次（全局 Context 单例）；
  3. model(input_ids, positions)——模型不带任何 attention 参数；
  4. lm_head 只在要出 token 的位置算。
池子大小由 warmup 量出来（权重 + 激活峰值都是实测）。"""
import torch

from config import Qwen3Config
from models.qwen3 import Qwen3ForCausalLM
from engine.kv_cache import KVCacheManager
from engine.context import set_context, reset_context
from engine.request import Request
from utils.loader import load_model


class ModelRunner:

    def __init__(self, model_dir: str, **engine_kwargs):
        self.config = Qwen3Config.from_pretrained(model_dir, **engine_kwargs)
        cfg = self.config
        self.device, self.dtype, self.max_seq_len = cfg.device, cfg.dtype, cfg.max_seq_len

        # ---- 1. 先建临时小池（供 warmup profile 用），模型接线到 manager ----
        self.kv = KVCacheManager(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
                                 cfg.block_size, max_num_blocks=256,
                                 dtype=cfg.dtype, device=cfg.device)
        self.model = Qwen3ForCausalLM(cfg, self.kv)
        load_model(self.model, model_dir, device=cfg.device, dtype=cfg.dtype,
                   tie_word_embeddings=cfg.tie_word_embeddings)
        # 此刻 memory_allocated = 权重 + 临时池，扣掉池才是纯权重
        weights_bytes = (torch.cuda.memory_allocated() - self.kv.pool_bytes()
                         if cfg.device == "cuda" else 0)

        # ---- 2+3. warmup 量激活峰值 → 重建真池 ----
        self._warmup_profile(weights_bytes)

    # ================= warmup 显存 profile =================

    def _warmup_profile(self, weights_bytes: int):
        """dummy 最大压力前向量激活峰值 → 算池子多大 → 重建池子。"""
        cfg = self.config
        B = max(2, min(cfg.batch_size, 8))
        from types import SimpleNamespace
        sp = SimpleNamespace(temperature=0.0, top_p=1.0, max_tokens=8, ignore_eos=True)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        reqs = [Request(10_000 + i, list(range(256)), sp, set()) for i in range(B)]
        for req in reqs:
            self.kv.allocate(req.id, 256)
        tokens = self.step(reqs, True, 0.0, 1.0)
        for req, t in zip(reqs, tokens.tolist()):
            req.step(t)
        for req in reqs:
            self.kv.ensure_append(req.id, req.cur_len)
        self.step(reqs, False, 0.0, 1.0)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        for r in reqs:
            self.kv.free(r.id)
        reset_context()

        # 峰值里含临时池，扣掉才是"权重 + 激活"的真预算
        peak_no_pool = peak - self.kv.pool_bytes()
        total = torch.cuda.get_device_properties(self.device).total_memory
        budget = int(total * cfg.gpu_memory_utilization) - peak_no_pool
        max_blocks = max(budget // self.kv.block_bytes, 1)
        self.kv.rebuild(max_blocks)

        gb = 1e9
        print(f"[v2 显存预算] 总显存 {total/gb:.1f} GB × 利用率 {cfg.gpu_memory_utilization}")
        print(f"  权重 {weights_bytes/gb:.2f} GB + 激活峰值 {(peak_no_pool - weights_bytes)/gb:.2f} GB (warmup 实测)")
        print(f"  → KV 池 {max_blocks} blocks × {cfg.block_size} tok "
              f"({self.kv.pool_bytes()/gb:.2f} GB, 每 block {self.kv.block_bytes/1e6:.2f} MB)")
        print(f"  容量: 1024-token 请求约 {max_blocks // self.kv.nblocks(1024)} 条并发 "
              f"(batch_size 上限仍为 {cfg.batch_size})")

    # ================= 平铺前向（唯一入口） =================

    @torch.no_grad()
    def step(self, reqs: list, is_prefill: bool,
             temperature: float, top_p: float) -> torch.Tensor:
        """这一步的请求摊平成一个 [total_q] 批：算好 Context → 前向 → 采样。
        返回 [len(reqs)] 个 token，与 reqs 对齐。"""
        neg = torch.finfo(self.dtype).min
        dev, bs = self.device, self.kv.block_size
        ids, poss, slots = [], [], []
        pf_spans, pf_masks, logits_idx = [], [], []
        dec_idx, dec_pos, dec_tabs_rows = [], [], []
        off = 0

        for req in reqs:
            if is_prefill:
                L = len(req.prompt_ids)
                pos = torch.arange(L)                       # prompt 位置 0..L-1
                ids.append(torch.tensor(req.prompt_ids))
                tabs = self.kv.block_table(req.id, self.kv.nblocks(L))
                causal = pos[:, None] >= pos[None, :]
                pf_masks.append(torch.where(causal, 0.0, neg).to(self.dtype).to(dev)[None, None])
                pf_spans.append((off, off + L, tabs, L))
                logits_idx.append(off + L - 1)             # 每行最后一个 prompt token
            else:
                p = req.cur_len - 1                         # 被喂 token 的位置
                pos = torch.tensor([p])
                ids.append(torch.tensor([req.generated[-1]]))
                dec_tabs_rows.append(self.kv.block_table(req.id, self.kv.nblocks(p + 1)))
                dec_idx.append(off)
                dec_pos.append(p)
                logits_idx.append(off)
            poss.append(pos)
            slots.append(pos // bs)                         # 先记块内序号，下面统一换槽位
            off += len(pos)

        # ---- decode 组：padded block 表 [n, nblk_max] + 长度 mask ----
        dec_n, dec_tabs, dec_read_len, dec_mask, dec_idx_t = 0, None, 0, None, None
        if dec_idx:
            dec_n = len(dec_idx)
            dec_read_len = max(dec_pos) + 1
            nblk = self.kv.nblocks(dec_read_len)
            dec_tabs = torch.stack([t if len(t) == nblk else torch.cat(
                [t, torch.zeros(nblk - len(t), dtype=torch.long, device=t.device)])
                for t in dec_tabs_rows])
            slots_d = torch.arange(dec_read_len)
            valid = slots_d[None, :] <= torch.tensor(dec_pos)[:, None]   # slot <= 自己的位置
            dec_mask = torch.where(valid[:, None, None], 0.0, neg).to(self.dtype).to(dev)
            dec_idx_t = torch.tensor(dec_idx, device=dev)

        # ---- 每 token 的全局槽位：逐请求查一次自己的 block 表 ----
        slot_list = []
        for j, req in enumerate(reqs):
            tab = pf_spans[j][2] if is_prefill else dec_tabs_rows[j]
            slot_list.append(tab[poss[j] // bs] * bs + poss[j] % bs)
        slot_mapping = torch.cat(slot_list).to(dev)

        input_ids = torch.cat(ids).to(dev)
        positions = torch.cat(poss).to(dev)
        # 上下文一次设好，全部层共享；模型只吃 (input_ids, positions)
        set_context(is_prefill=is_prefill, slot_mapping=slot_mapping,
                   dec_n=dec_n, dec_idx=dec_idx_t, dec_tabs=dec_tabs,
                   dec_read_len=dec_read_len, dec_mask=dec_mask,
                   pf_spans=pf_spans, pf_masks=pf_masks)

        hidden = self.model(input_ids, positions)                 # [total_q, hidden]
        h_need = hidden[torch.tensor(logits_idx, device=dev)]    # 只在要出 token 的位置算 lm_head
        logits = self.model.compute_logits(h_need)
        return self.sample(logits, temperature, top_p)

    # ================= 采样 =================

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
