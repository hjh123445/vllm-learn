"""v6.3 阶段二验证：MLA / HyperConnection / HCA 压缩器的单元对拍。

对拍对象：HF transformers 5.12 的 DeepseekV4 组件（同权重同输入逐位对比）。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.dsv4_components import (
    HyperConnection, HCACompressor, mla_attention, apply_partial_rope)

MODEL = "/data/xmhuang6/deepseek/DeepSeek-V4-Flash"


def test_hyper_connection():
    """与 HF DeepseekV4HyperConnection 逐位对拍。"""
    sys.path.insert(0, "/data/ifly_jyhe23/hot_model_top5/audio_understand/qwen2-audio/transformers-5.12.0/src")
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection
    from types import SimpleNamespace

    cfg = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
                          rms_norm_eps=1e-6, hidden_size=512)
    torch.manual_seed(0)
    hf_hc = DeepseekV4HyperConnection(cfg).float().eval()
    # 同权重构造我们的
    our_hc = HyperConnection(hc_mult=4, hidden=512, sinkhorn_iters=20).float().eval()
    our_hc.fn.data = hf_hc.fn.data.clone()
    our_hc.base.data = hf_hc.base.data.clone()
    our_hc.scale.data = hf_hc.scale.data.clone()

    streams = torch.randn(6, 4, 512)   # [T, hc, hidden]
    with torch.no_grad():
        hf_post, hf_comb, hf_collapsed = hf_hc(streams.unsqueeze(0))  # HF 带 batch 维
        our_post, our_comb, our_collapsed = our_hc(streams)

    print(f"HC post diff:    {(hf_post.squeeze(0) - our_post).abs().max().item():.2e}")
    print(f"HC comb diff:    {(hf_comb.squeeze(0) - our_comb).abs().max().item():.2e}")
    print(f"HC collapsed diff: {(hf_collapsed.squeeze(0) - our_collapsed).abs().max().item():.2e}")
    d = (hf_collapsed.squeeze(0) - our_collapsed).abs().max().item()
    assert d < 1e-4, f"HyperConnection 对拍失败 {d}"
    print("HyperConnection: PASS\n")


def test_hca_compressor():
    """与 HF DeepseekV4HCACompressor 对拍（无 cache 的单发模式）。"""
    sys.path.insert(0, "/data/ifly_jyhe23/hot_model_top5/audio_understand/qwen2-audio/transformers-5.12.0/src")
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HCACompressor
    from types import SimpleNamespace

    cfg = SimpleNamespace(hidden_size=512, head_dim=128,
                          compress_rates={"heavily_compressed_attention": 32},
                          rms_norm_eps=1e-6,
                          max_position_embeddings=4096, rope_theta=10000,
                          rope_parameters={}, rope_scaling=None)
    torch.manual_seed(1)
    hf_c = DeepseekV4HCACompressor(cfg).float().eval()
    # HF 的 rope 初始化依赖完整 config（yarn 等）——对拍只验证压缩数学，
    # monkeypatch 跳过 rope（我们的实现里 rope 是可选参数，默认跳过）
    import torch.nn as _nn
    class _NoRope(_nn.Module):
        def forward(self, x, position_ids=None, layer_type=None):
            shape = x.shape[:-1] if x.dim() > 2 else (x.shape[0],)
            return torch.ones(*shape, 16), torch.zeros(*shape, 16)
    hf_c.rotary_emb = _NoRope()
    our_c = HCACompressor(hidden=512, head_dim=128, compress_rate=32).float().eval()
    our_c.kv_proj.weight.data = hf_c.kv_proj.weight.data.clone()
    our_c.gate_proj.weight.data = hf_c.gate_proj.weight.data.clone()
    our_c.position_bias.data = hf_c.position_bias.data.clone()
    our_c.norm.weight.data = hf_c.kv_norm.weight.data.clone()

    hidden_t = torch.randn(64, 512)      # 平铺 [T, H]（我们的约定）
    hidden = hidden_t.unsqueeze(0)        # HF 的 [B=1, T, H]
    pos_ids = torch.arange(64)[None]     # [B=1, T]
    with torch.no_grad():
        hf_out, _ = hf_c(hidden, None, pos_ids, None, 0)
        our_out = our_c(hidden_t)
    # HF 返回 [B, 1, T', D] → [T', D]
    d = (hf_out.squeeze(0).squeeze(0) - our_out).abs().max().item()
    print(f"HCA 压缩器 diff: {d:.2e}")
    assert d < 1e-4, f"HCA 对拍失败 {d}"
    print(f"HCA 压缩器: PASS（64 tok → {our_out.shape[0]} 条压缩 KV）\n")


def test_mla_math():
    """MLA 核心数学自检：sink 只吸收概率（概率守恒）+ partial rope 维度。"""
    torch.manual_seed(2)
    T, L, H, D = 5, 20, 8, 64
    q = torch.randn(T, H, D)
    kv = torch.randn(L, D)
    sink = torch.randn(H) * 2

    out = mla_attention(q, kv, sink, D ** -0.5)
    assert out.shape == (T, H, D)
    # 无 rope 时 cos/sin 为 None——输出直接是 V 的加权
    # 自检: sink → -inf 时应退化为普通 attention
    out_nsink = mla_attention(q, kv, torch.full((H,), -1e9), D ** -0.5)
    scores = torch.einsum("thd,ld->htl", q, kv) * D ** -0.5
    ref = torch.einsum("htl,ld->thd", scores.softmax(-1), kv)
    d = (out_nsink - ref).abs().max().item()
    print(f"MLA 无 sink 退化对拍: {d:.2e}")
    assert d < 1e-5

    # partial rope（interleaved + 尾部 rope 维）: 前 D-32 维不变、尾 32 维旋转
    cos = torch.randn(T, 16)
    sin = torch.randn(T, 16)
    x = torch.randn(T, H, D)
    xr = apply_partial_rope(x, cos, sin, rope_dim=32)
    assert xr[..., :-32].equal(x[..., :-32]), "nope 区（头部）必须不变"
    assert not xr[..., -32:].equal(x[..., -32:])
    print("MLA partial rope 维度语义: PASS")


if __name__ == "__main__":
    print("===== v6.3 阶段二：组件对拍 =====")
    test_hyper_connection()
    test_hca_compressor()
    test_mla_math()
    print("ALL PASS")
