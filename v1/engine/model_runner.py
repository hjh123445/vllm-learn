# v1/engine/model_runner.py
"""v1 ModelRunner：权重、前向、attention mask、采样。
prefill 逐请求做（batch=1，无需 padding），decode 把当前全部活跃请求合成一个 batch。"""
import torch

from config import Qwen3Config
from models.qwen3 import Qwen3ForCausalLM
from utils.loader import load_model


class ModelRunner:

    def __init__(self, model_dir: str, **engine_kwargs):
        self.config = Qwen3Config.from_pretrained(model_dir, **engine_kwargs)
        self.device = self.config.device
        self.dtype = self.config.dtype
        self.max_seq_len = self.config.max_seq_len

        self.model = Qwen3ForCausalLM(self.config)
        load_model(self.model, model_dir,
                   device=self.device, dtype=self.dtype,
                   tie_word_embeddings=self.config.tie_word_embeddings)

    # ---------------- 前向 ----------------

    @torch.no_grad()
    def forward(self, input_ids, positions, cache_rows, write_pos, attn_mask):
        """返回 [b, vocab]（最后位置 logits）。"""
        b, s = input_ids.shape
        assert b <= self.config.batch_size
        hidden = self.model(input_ids.to(self.device),
                            positions.to(self.device),
                            cache_rows.to(self.device),
                            write_pos.to(self.device),
                            attn_mask.to(self.device))
        return self.model.compute_logits(hidden[:, -1])

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

    # ---------------- prefill（单请求） ----------------

    @torch.no_grad()
    def prefill(self, req) -> torch.Tensor:
        """单条请求的 prefill：batch=1，从位置 0 连续写入，不需要 padding mask。"""
        L = len(req.prompt_ids)
        neg = torch.finfo(self.dtype).min
        input_ids = torch.tensor([req.prompt_ids])                    # [1, L]
        positions = torch.arange(L)[None]                            # [1, L]
        cache_rows = torch.tensor([req.slot])                        # [1]
        write_pos = torch.zeros(1, dtype=torch.long)                 # [1]
        causal = torch.ones(L, L, dtype=torch.bool).tril()
        attn_mask = torch.where(causal, 0.0, neg).to(self.dtype)[None, None]  # [1,1,L,L]
        return self.forward(input_ids, positions, cache_rows, write_pos, attn_mask)

    # ---------------- decode（当前活跃请求合批） ----------------

    @torch.no_grad()
    def decode_step(self, reqs: list, temperature: float, top_p: float) -> torch.Tensor:
        """所有 running 请求各出一个 token：组一个变长的 batch（无 padding，靠 mask 区分长度）。"""
        neg = torch.finfo(self.dtype).min
        b = len(reqs)
        tokens = torch.tensor([[r.generated[-1]] for r in reqs])    # [b, 1]
        positions = torch.tensor([[r.cur_len - 1] for r in reqs])   # [b, 1] 新 token 的位置
        cache_rows = torch.tensor([r.slot for r in reqs])            # [b]
        write_pos = positions.flatten()                               # [b] 写入位置 = 当前位置
        read_len = int(write_pos.max()) + 1                           # 读到最长请求的新 token
        slots = torch.arange(read_len)
        valid = slots[None, :] <= write_pos[:, None]                  # [b, read_len] 只看到自己的长度
        attn_mask = torch.where(valid[:, None, None], 0.0, neg).to(self.dtype)  # [b,1,1,read_len]
        logits = self.forward(tokens, positions, cache_rows, write_pos, attn_mask)
        return self.sample(logits, temperature, top_p)
