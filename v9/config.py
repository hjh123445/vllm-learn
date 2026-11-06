import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = True
    # "batched" maximizes GEMM fusion; "per_request" preserves the
    # single-request shapes for reproducibility at a throughput cost.
    batch_execution_mode: str = "batched"
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    max_kvcache_blocks: int | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        if self.max_kvcache_blocks is not None and self.max_kvcache_blocks <= 0:
            raise ValueError('max_kvcache_blocks must be positive')
        if self.batch_execution_mode not in ("batched", "per_request"):
            raise ValueError("batch_execution_mode must be batched or per_request")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 多模态/Omni 模型的 max_position_embeddings 在 thinker.text_config 里
        mpe = getattr(self.hf_config, "max_position_embeddings", None)
        if mpe is None:
            tc = getattr(self.hf_config, "thinker_config", None) or {}
            ttc = (tc.get("text_config", {}) if isinstance(tc, dict)
                   else getattr(tc, "text_config", None))
            if isinstance(ttc, dict):
                mpe = ttc.get("max_position_embeddings", 32768)
            else:
                mpe = getattr(ttc, "max_position_embeddings", 32768)
        self.max_model_len = min(self.max_model_len, mpe)
