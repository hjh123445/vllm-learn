"""Device-native RoPE cache must agree with the model's HF text positions."""
import sys
from pathlib import Path
import torch
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextRotaryEmbedding
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from layers.rotary_embedding import RotaryEmbeddding

model=sys.argv[1]
torch.set_num_threads(4)
cfg=AutoConfig.from_pretrained(model).thinker_config.text_config
for device in ['cpu','cuda']:
    for dtype in [torch.float32,torch.bfloat16]:
        rope=RotaryEmbeddding(8192,cfg.head_dim,cfg.rope_parameters['rope_theta']).to(device=device,dtype=dtype)
        ref=Qwen3OmniMoeThinkerTextRotaryEmbedding(cfg).to(device=device)
        positions=torch.tensor([[0,1,255,511,512,1026,4095,8191]],device=device)
        cos,sin=ref(torch.zeros(1,dtype=dtype,device=device),positions)
        c,s=rope.cos_sin_cache[positions[0],0].chunk(2,-1)
        torch.testing.assert_close(torch.cat((c,c),-1),cos[0],atol=0,rtol=0)
        torch.testing.assert_close(torch.cat((s,s),-1),sin[0],atol=0,rtol=0)
        print(device,dtype,'HF text RoPE exact',flush=True)
