"""RoPE must match HF text positions without a checkpoint or import side effects."""
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layers.rotary_embedding import RotaryEmbeddding


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_device_rope_matches_hf(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = AutoConfig.for_model("qwen3_omni_moe", thinker_config={"text_config": {
        "hidden_size": 256, "head_dim": 64, "num_attention_heads": 4,
        "num_key_value_heads": 2, "rope_theta": 1000000.,
        "max_position_embeddings": 8192,
    }}).thinker_config.text_config
    rope = RotaryEmbeddding(8192, cfg.head_dim, cfg.rope_parameters["rope_theta"])
    rope = rope.to(device=device, dtype=dtype)
    ref = Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg).to(device=device)
    positions = torch.tensor([[0, 1, 255, 511, 512, 1026, 4095, 8191]], device=device)
    cos, sin = ref(torch.zeros(1, dtype=dtype, device=device), positions)
    c, s = rope.cos_sin_cache[positions[0], 0].chunk(2, -1)
    torch.testing.assert_close(torch.cat((c, c), -1), cos[0], atol=0, rtol=0)
    torch.testing.assert_close(torch.cat((s, s), -1), sin[0], atol=0, rtol=0)
