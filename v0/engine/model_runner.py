# engine/model_runner.py
"""显存与前向的唯一管理者：权重、KV cache、logits、采样都只在这里出现。"""
import torch

from config import Qwen3Config
from models.qwen3 import Qwen3ForCausalLM
from utils.loader import load_model


class ModelRunner:

    def __init__(self, model_dir: str, **engine_kwargs):
        # engine_kwargs: batch_size / max_seq_len / device / dtype / max_tokens / eos
        self.config = Qwen3Config.from_pretrained(model_dir, **engine_kwargs)
        self.device = self.config.device
        self.dtype = self.config.dtype
        self.max_seq_len = self.config.max_seq_len

        self.model = Qwen3ForCausalLM(self.config)
        # 内部完成：CPU装载+对账 → tie embeddings → .to(device,dtype) → .eval()
        load_model(self.model, model_dir,
                   device=self.device, dtype=self.dtype,
                   tie_word_embeddings=self.config.tie_word_embeddings)

    # ---------------- cache 管理 ----------------

    def reset_cache(self):
        """每个新 batch 调一次（cur_len 清零，数据靠覆盖）"""
        for m in self.model.modules():
            if hasattr(m, "cache"):
                m.cache.reset()

    @property
    def cur_len(self) -> int:
        for m in self.model.modules():
            if hasattr(m, "cache"):
                return m.cache.cur_len
        return 0

    # ---------------- 前向 ----------------

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor,
                attn_mask: torch.Tensor | None) -> torch.Tensor:
        """
        input_ids : [b, s]  prefill=整个prompt / decode=[b,1]
        positions : RoPE真实位置（左padding下每请求各算各的）
        attn_mask : float mask，可见=0 不可见=-inf，形状 [b,1,s_q,cur_len] 或 None
        返回       : [b, vocab] 只有最后一个位置的 logits
        """
        b, s = input_ids.shape
        assert b <= self.config.batch_size
        assert self.cur_len + s <= self.max_seq_len, \
            f"cache容量不足: {self.cur_len}+{s} > {self.max_seq_len}"

        hidden = self.model(input_ids.to(self.device),
                            positions.to(self.device),
                            None if attn_mask is None else attn_mask.to(self.device))
        return self.model.compute_logits(hidden[:, -1])   # [b, vocab]，只算最后位置

    # ---------------- 采样 ----------------

    @torch.no_grad()
    def sample(self, logits: torch.Tensor,
               temperature: float, top_p: float) -> torch.Tensor:
        """[b, vocab] → [b]；temperature<=0 为贪心（验证时用它）"""
        if temperature <= 0:
            return logits.argmax(-1)
        probs = (logits / temperature).softmax(-1)
        if top_p < 1.0:                                    # 核采样
            sorted_probs, sorted_idx = probs.sort(-1, descending=True)
            cum = sorted_probs.cumsum(-1)
            sorted_probs[cum - sorted_probs > top_p] = 0    # 越过阈值的尾部截断
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
            probs = probs / probs.sum(-1, keepdim=True)
        return torch.multinomial(probs, 1).squeeze(-1)