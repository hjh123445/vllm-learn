# v9/layers/dequant_gemm.py
"""v9.2 triton 融合 dequant-GEMM：int4 打包权重在寄存器里反量化，HBM 只读 int4。

B1 的问题（DESIGN §II.2 实测）：unpack→bf16 写回→再读，HBM 流量
= int4 读 + bf16 写 + bf16 读 > 纯 bf16 读一遍 → decode 慢 5.5x。
B2 的解法：GEMM tile kernel 里 load uint8 → 位运算拆 nibble → ×scale →
tl.dot——**中间值不落 HBM**，权重读流量真实降为 1/3.9。

v3 kernel（正确处理 interleave 的关键技巧）：
  打包布局 [N, K/2] 的每个 uint8 = (偶列 q0 | 奇列 q1<<4)。
  数学恒等式：Y = Σ_k x[k]·W[n,k]
    = Σ_j ( x[2j]·W[n,2j] + x[2j+1]·W[n,2j+1] )      ← 按偶/奇列配对
  所以 kernel 只需把 x 也按偶/奇列拆开加载（各 [BM, BK/2]），W 的一个
  uint8 tile 直接拆出两个 bf16 半块，各做一次 dot 累加——**无需 interleave
  重建**（v2 版 tl.join 的浪费源）。scale 按 group 预载为 [BN, BK] fp32 行
  向量代价高，改为每半块一次性算好 s_g 的展开（group=128 时 BK=64 内恰
  一组，s 是标量 per (n, tile)）。

数值纪律：acc fp32 累加（v4 kernel 同款）。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_gemm_kernel(X, WPK, WS, Y,
                         M, N, K,
                         stride_xm, stride_wn, stride_ym,
                         G: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Y = X @ dequant(W).T
    X: [M, K] bf16; WPK: [N, K/2] uint8; WS: [N, K/G] fp16。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)            # 全 K tile 的列
    rk2 = tl.arange(0, BK // 2)      # packed/偶奇半块的列

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, K, BK):
        # ---- W tile: uint8 [BN, BK/2] → 偶/奇两个 bf16 半块 ----
        wp = tl.load(WPK + rn[:, None].to(tl.int64) * (K // 2) + k0 // 2 + rk2[None, :],
                     mask=(rn[:, None] < N), other=0)               # uint8
        w_e = (wp & 0xF).to(tl.float32)                             # 偶列 (n, j)
        w_e = tl.where(w_e >= 8.0, w_e - 16.0, w_e)
        w_o = (wp >> 4).to(tl.float32)                              # 奇列 (n, j)
        w_o = tl.where(w_o >= 8.0, w_o - 16.0, w_o)

        # ---- group scale（BK=64, G=128: 每 tile 半组，逐列索引）----
        gs_e = (k0 + 2 * rk2) // G
        gs_o = (k0 + 2 * rk2 + 1) // G
        s_base = WS + rn[:, None].to(tl.int64) * (K // G)
        s_e = tl.load(s_base + gs_e[None, :], mask=(rn[:, None] < N), other=0.0).to(tl.float32)
        s_o = tl.load(s_base + gs_o[None, :], mask=(rn[:, None] < N), other=0.0).to(tl.float32)
        w_e, w_o = w_e * s_e, w_o * s_o

        # ---- x tile 按偶/奇列拆开 ----
        x_base = X + rm[:, None].to(tl.int64) * stride_xm
        x_e = tl.load(x_base + (k0 + 2 * rk2)[None, :],
                      mask=(rm[:, None] < M), other=0.0)            # [BM, BK/2]
        x_o = tl.load(x_base + (k0 + 2 * rk2 + 1)[None, :],
                      mask=(rm[:, None] < M), other=0.0)

        we_t = tl.trans(w_e.to(tl.bfloat16))
        wo_t = tl.trans(w_o.to(tl.bfloat16))
        acc += tl.dot(x_e.to(tl.bfloat16), we_t, input_precision="ieee")
        acc += tl.dot(x_o.to(tl.bfloat16), wo_t, input_precision="ieee")

    y_ptrs = Y + rm[:, None].to(tl.int64) * stride_ym + rn[None, :]
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def dequant_gemm(x: torch.Tensor, w_packed: torch.Tensor, w_scales: torch.Tensor,
                 group: int = 128) -> torch.Tensor:
    """x [M, K] bf16 × dequant(W) [N, K] → [M, N]。
    w_packed [N, K/2] uint8; w_scales [N, K/G] fp16。
    约束: K % (2*64) == 0（BK=64 对齐打包半块）；G % BK == 0（scale tile 内单一组）。"""
    M, K = x.shape
    N = w_packed.shape[0]
    assert K % 2 == 0 and K % group == 0
    assert group % 64 == 0, f"G={group} 需 64 的倍数（BK tile 不跨组）"
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    # decode 形状实测最优（A800 扫描 BM×BN×BK×warps）: 0.042ms @ M=8
    BM, BN, BK = 8, 32, 128
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _dequant_gemm_kernel[grid](
        x, w_packed, w_scales, y,
        M, N, K, x.stride(0), w_packed.stride(0), y.stride(0),
        G=group, BM=BM, BN=BN, BK=BK, num_warps=8)
    return y


def quant_linear(x, w_packed, w_scales, group=128, dequant_threshold=64):
    """量化权重的混合路由（vLLM w4a16 的策略同款）：
    - 小 M（decode，带宽 bound）：B2 融合 kernel——int4 读 + 寄存器反量化；
    - 大 M（prefill，算力 bound）：B1 反量化 + cuBLAS——dequant 开销被
      大 M 摊薄（实测 M=4096: B1 0.356ms vs B2 4.4ms，B1 赢 12x）。
    阈值由 K/N 形状微调，默认 64（M<64 走 kernel）。"""
    M = x.shape[0]
    if M <= dequant_threshold:
        return dequant_gemm(x, w_packed, w_scales, group)
    from layers.quantized import dequantize
    w = dequantize(w_packed, w_scales, group).to(x.dtype)
    return x @ w.t()


if __name__ == "__main__":
    import time
    from layers.quantized import rtn_quantize, dequantize

    torch.manual_seed(0)
    N, K = 2048, 2048
    w = (torch.randn(N, K) * 0.02).to(torch.bfloat16)
    packed, scales = rtn_quantize(w.cpu(), 128)
    packed, scales = packed.cuda(), scales.cuda()

    for M, tag in [(8, "decode bs=8"), (4096, "prefill 4096")]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        ref = (x.float() @ dequantize(packed, scales, 128).cuda().t())
        out = dequant_gemm(x, packed, scales, 128).float()
        rel = (out - ref).abs().max().item() / ref.abs().max().item()
        print(f"M={M:<5}({tag:<13}) 数值 rel {rel:.2e}", end=" ")
        assert rel < 1e-2

        w_bf = dequantize(packed, scales, 128).to(torch.bfloat16).cuda()

        def bench(fn, n=50):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.time() - t0) / n * 1000

        t_bf = bench(lambda: x @ w_bf.t())
        t_k = bench(lambda: dequant_gemm(x, packed, scales, 128))
        print(f"| bf16 {t_bf:.3f}ms vs B2 {t_k:.3f}ms (x{t_k/t_bf:.2f})")
    print("\nALL PASS" )
