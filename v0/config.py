# config.py
import json, os
from dataclasses import dataclass, fields
import torch


@dataclass
class Qwen3Config:
    # ---- 模型结构：字段名与 config.json 完全一致 ----
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"
    # ---- 引擎参数（config.json 里没有，从 from_pretrained 的 engine_kwargs 传）----
    batch_size: int = 8
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    max_tokens: int = 256
    eos: int = 151645
    max_seq_len: int = 4096

    @classmethod
    def from_pretrained(cls, model_dir: str, **engine_kwargs) -> "Qwen3Config":
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
        known = {f.name for f in fields(cls)}
        mapped = {k: v for k, v in raw.items() if k in known}   # 名字一致，不需要翻译表
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if k in known}
        return cls(**{**mapped, **engine_kwargs})