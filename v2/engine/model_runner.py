# v2/engine/model_runner.py
"""v2 ModelRunner：显存的唯一所有者。

与 v1 的本质区别——v1 池子 = batch_size × max_seq_len 拍脑袋预分配；
v2 分三步：
  1. 载权重，量出权重实占（memory_allocated）；
  2. warmup profile：建临时小池，跑一次 prefill + decode 的最大压力前向，
     量出激活峰值（max_memory_allocated）；
  3. 池子大小 = (总显存 × gpu_memory_utilization − 权重 − 激活峰值) ÷ 每 block 字节
全部是量出来的，不是算出来的。
"""
import torch

from config import Qwen3Config
from models.qwen3 import Qwen3ForCausalLM
from engine.kv_cache import KVCacheManager
from utils.loader import load_model


class ModelRunner:

    def __init__(self, model_dir: str, **engine_kwargs):
        self.config = Qwen3Config.from_pretrained(model_dir, **engine_kwargs)
        cfg = self.config
        self.device, self.dtype, self.max_seq_len = cfg.device, cfg.dtype, cfg.max_seq_len

        # ---- 1. 先建临时小池（供 warmup profile 用），模型接线到 manager ----
        # 临时池只要够 dummy 压测用（8 请求 × 256 token ≈ 128 block），量完立刻释放
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
        """最大压力 dummy 前向量激活峰值 → 算出池子该多大 → 重建池子。
        dummy 覆盖 prefill 和 decode 两种形状：prefill 一次性吃 256 token，
        decode 是 LM head 全 vocab 前向 + 逐层 gather 的峰值形状。"""
        cfg = self.config
        B = max(2, min(cfg.batch_size, 8))          # 压测用的假 batch
        from engine.scheduler import Request
        sp = type("SP", (), {"temperature": 0.0, "top_p": 1.0,
                             "max_tokens": 8, "ignore_eos": True})()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        reqs = []
        for i in range(B):
            req = Request(10_000 + i, list(range(256)), sp, eos_ids=set())
            self.kv.allocate(req.id, 256)
            reqs.append(req)
            self.prefill(req)                        # prefill 峰值
            req.generated = [0]                     # 假 token，让 decode_step 有东西可喂
        for req in reqs:
            self.kv.ensure_append(req.id, req.cur_len)   # 跨 block 边界的补块也要算进峰值
        self.decode_step(reqs, 0.0, 1.0)             # decode 峰值
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        for r in reqs:
            self.kv.free(r.id)

        # 峰值里含临时池，扣掉它才是"权重 + 激活"的真预算
        peak_no_pool = peak - self.kv.pool_bytes()
        total = torch.cuda.get_device_properties(self.device).total_memory
        budget = int(total * cfg.gpu_memory_utilization) - peak_no_pool
        max_blocks = max(budget // self.kv.block_bytes, 1)

        self.kv.rebuild(max_blocks)

        # 报告：全部是量出来的数
        act = peak_no_pool - weights_bytes
        gb = 1e9
        print(f"[v2 显存预算] 总显存 {total/gb:.1f} GB × 利用率 {cfg.gpu_memory_utilization}")
        print(f"  权重 {weights_bytes/gb:.2f} GB + 激活峰值 {act/gb:.2f} GB (warmup 实测)")
        print(f"  → KV 池 {max_blocks} blocks × {cfg.block_size} tok "
              f"({self.kv.pool_bytes()/gb:.2f} GB, 每 block {self.kv.block_bytes/1e6:.2f} MB)")
        n_1k = max_blocks // self.kv.nblocks(1024)
        print(f"  容量: 1024-token 请求约 {n_1k} 条并发 (batch_size 上限仍为 {cfg.batch_size})")

    # ================= 前向 =================

    @torch.no_grad()
    def forward(self, input_ids, positions, tabs, attn_mask) -> torch.Tensor:
        """返回 [b, vocab]（最后位置 logits）。"""
        hidden = self.model(input_ids.to(self.device),
                            positions.to(self.device),
                            tabs.to(self.device),
                            attn_mask.to(self.device))
        return self.model.compute_logits(hidden[:, -1])

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

    # ================= prefill（单请求，无 padding） =================

    @torch.no_grad()
    def prefill(self, req) -> torch.Tensor:
        L = len(req.prompt_ids)
        neg = torch.finfo(self.dtype).min
        input_ids = torch.tensor([req.prompt_ids])                  # [1, L]
        positions = torch.arange(L)[None]                            # [1, L]
        tabs = self.kv.tables_batch([req.id], self.kv.nblocks(L))   # [1, nblk]
        causal = torch.ones(L, L, dtype=torch.bool).tril()
        attn_mask = torch.where(causal, 0.0, neg).to(self.dtype)[None, None]  # [1,1,L,L]
        return self.forward(input_ids, positions, tabs, attn_mask)

    # ================= decode（活跃请求合批） =================

    @torch.no_grad()
    def decode_step(self, reqs: list, temperature: float, top_p: float) -> torch.Tensor:
        neg = torch.finfo(self.dtype).min
        b = len(reqs)
        tokens = torch.tensor([[r.generated[-1]] for r in reqs])    # [b, 1]
        pos = torch.tensor([r.cur_len - 1 for r in reqs])           # [b] 被喂 token 的位置
        positions = pos[:, None]                                     # [b, 1]
        read_len = int(pos.max()) + 1                                # 最长请求读多长
        tabs = self.kv.tables_batch([r.id for r in reqs],
                                    self.kv.nblocks(read_len))      # [b, nblk]
        slots = torch.arange(read_len)
        valid = slots[None, :] <= pos[:, None]                      # [b, read_len] 只见自己写过的
        attn_mask = torch.where(valid[:, None, None], 0.0, neg).to(self.dtype)  # [b,1,1,read_len]
        logits = self.forward(tokens, positions, tabs, attn_mask)
        return self.sample(logits, temperature, top_p)
