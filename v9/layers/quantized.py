# v9/layers/quantized.py
"""v9.1 weight-only 量化组件：int4 打包存储 + RTN 离线量化器 + QuantizedLinear。

设计依据（DESIGN.md §II.2 的实测数据）：
- 粒度选 per-group g=128（AWQ/GPTQ 标准）：int4 相对 MSE ~1.5e-2，
  g=32 只再降 1/3 但 scale 开销翻 4 倍（有效位宽 4.5 vs 4.125 bit）；
- Path B1（forward unpack→bf16 matmul）实测 decode 慢 5.5x——它的角色是
  【正确性基线 + 显存收益】，速度收益在 v9.2 的 triton 融合 kernel；
- 打包格式：uint8 双 nibble（低半=偶数位，高半=奇数位），K 必须 2|K 且 g|K
  （不满足 pad，pad 列 scale 置 1、权重 0——不影响数值）。

与并行切分的相容性（DESIGN.md §II.3.2）：量化粒度沿输入维（K），TP 切的是
输出维（q/o/w13）或专家维（EP）——切分后再量化天然对齐；w2 的 TP 切 K 维
会切断 group，本版 w2 不做 TP（量化版引擎单卡/EP 跑，TP 留 v9.4）。
"""

import torch
import torch.nn.functional as F
from torch import nn

QMAX_INT4 = 7          # 对称量化 [-8, 7]


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """int4 张量 [.., K] → uint8 [.., K/2]（双 nibble）。q 取值 [-8,7]。"""
    assert q.shape[-1] % 2 == 0
    q = q.to(torch.int8)
    lo, hi = q[..., 0::2], q[..., 1::2]
    return (lo.to(torch.uint8) & 0xF) | (hi.to(torch.uint8) << 4)


def unpack_int4(packed: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """uint8 [.., K/2] → int4 值（带符号扩展）[.., K]。"""
    lo = (packed & 0xF).to(torch.int8)
    hi = (packed >> 4).to(torch.int8)
    # 高 nibble 的 8-15 是负数（int4 补码）——位运算保持
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    out = torch.stack([lo, hi], dim=-1).flatten(-2)
    return out.to(dtype)


def rtn_quantize(w: torch.Tensor, group: int = 128):
    """对称 RTN 量化（沿输入维 K 分组）。

    返回 (packed [N, K/2] uint8, scales [N, K/g] fp16)。
    pad 处理：K 不整除 g 时最后一组补零（scale 正常算，数值不受影响）。"""
    N, K = w.shape
    assert K % 2 == 0, f"K={K} 必须偶数（双 nibble 打包）"
    pad = (-K) % group
    wp = F.pad(w.float(), (0, pad)) if pad else w.float()
    wv = wp.reshape(N, -1, group)
    s = wv.abs().amax(-1).clamp(min=1e-10) / QMAX_INT4        # [N, n_groups]
    q = (wv / s.unsqueeze(-1)).round().clamp_(-8, QMAX_INT4)
    packed = pack_int4(q.reshape(N, -1))                      # [N, K'/2]
    return packed, s.to(torch.float16)


def dequantize(packed: torch.Tensor, scales: torch.Tensor, group: int = 128) -> torch.Tensor:
    """反量化（Path A：装载时全量转 bf16 的数学参考）。"""
    q = unpack_int4(packed, torch.float32)                    # [N, K]
    N, K = q.shape
    pad = (-K) % group
    if pad:
        q = F.pad(q, (0, pad))
    sv = scales.float().unsqueeze(-1)                         # [N, n_groups, 1]
    return (q.reshape(N, -1, group) * sv).reshape(N, -1)[:, :K]


class QuantizedLinear(nn.Module):
    """Path B1：int4 打包权重 + forward 时 unpack→dequant→bf16 matmul。

    显存收益即时（权重 -75%），速度收益为负（DESIGN §II.2 实测 decode 慢 5.5x，
  unpack 的中间读写流量 > 省下的权重读流量）——它是量化引擎的正确性基线，
    v9.2 的 triton 融合 dequant-GEMM 替换其 forward 后速度收益才兑现。

    weight_loader 协议兼容：装载 bf16 safetensors 时在线 RTN（引擎无需
    离线量化步骤即可跑量化模式——离线量化器 v9.2 再做成独立产物）。"""

    def __init__(self, in_dim: int, out_dim: int, group: int = 128):
        super().__init__()
        assert in_dim % 2 == 0
        self.in_dim, self.out_dim, self.group = in_dim, out_dim, group
        # 打包存储 [N, K/2] uint8 + scales [N, K/g] fp16
        self.register_buffer("w_packed", torch.zeros(out_dim, (in_dim + 1) // 2, dtype=torch.uint8))
        self.register_buffer("w_scales", torch.zeros(out_dim, (in_dim + group - 1) // group,
                                                     dtype=torch.float16))

    def weight_loader(self, param, loaded, shard_id=None):
        """在线量化装载：bf16 权重 → RTN → 打包进 buffer。"""
        w = loaded.to(torch.float32)
        if w.shape[-1] % self.group != 0:
            pass                                       # rtn_quantize 内部 pad
        packed, scales = rtn_quantize(w, self.group)
        self.w_packed.copy_(packed.to(self.w_packed.device))
        self.w_scales.copy_(scales.to(self.w_scales.device))

    def dequant_weight(self) -> torch.Tensor:
        """bf16 反量化权重（调试/对拍用）。"""
        return dequantize(self.w_packed, self.w_scales, self.group).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """B1 路径：unpack → group dequant → bf16 matmul。
        x: [T, K] → [T, N]。"""
        T, K = x.shape
        assert K == self.in_dim
        q = unpack_int4(self.w_packed, x.dtype)                # [N, K] bf16
        N = q.shape[0]
        pad = (-K) % self.group
        if pad:
            q = F.pad(q, (0, pad))
        sv = self.w_scales.to(x.dtype).unsqueeze(-1)          # [N, n_groups, 1]
        deq = (q.reshape(N, -1, self.group) * sv).reshape(N, -1)[:, :K]
        return F.linear(x, deq)

    @staticmethod
    def from_bf16(linear: nn.Module, group: int = 128) -> "QuantizedLinear":
        """把已装载 bf16 的 BaseLinear 就地转换（保持 weight_loader 协议）。"""
        ql = QuantizedLinear(linear.weight.shape[1], linear.weight.shape[0], group)
        ql.weight_loader(None, linear.weight.data)
        return ql


if __name__ == "__main__":
    # 单元自测：数值 + 打包往返
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.03
    packed, scales = rtn_quantize(w)
    rec = dequantize(packed, scales)
    rel_mse = ((rec - w) ** 2).mean() / (w ** 2).mean()
    print(f"RTN g=128 相对 MSE: {rel_mse:.2e}（期望 ~1.5e-2 量级）")
    assert rel_mse < 5e-2

    # QuantizedLinear 前向数值一致性（vs 反量化权重的 F.linear）
    ql = QuantizedLinear(512, 256)
    ql.weight_loader(None, w)
    x = torch.randn(8, 512)
    out_ql = ql(x)
    out_ref = F.linear(x, ql.dequant_weight().float())
    diff = (out_ql.float() - out_ref).abs().max().item()
    print(f"QuantizedLinear vs 反量化参考 max diff: {diff:.2e}（期望 <1e-2，纯 bf16 舍入）")
    assert diff < 1e-2
    # 显存收益
    n_bf = w.numel() * 2
    n_q = ql.w_packed.numel() + ql.w_scales.numel() * 2
    print(f"显存: bf16 {n_bf/1e3:.0f}KB → int4+scale {n_q/1e3:.0f}KB ({n_bf/n_q:.2f}x)")
    print("ALL PASS")
