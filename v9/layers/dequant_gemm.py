# v9/layers/dequant_gemm.py
"""v9.2 triton 融合 dequant-GEMM：int4 打包权重在寄存器里反量化，HBM 只读 int4。

B1 的问题（DESIGN §II.2 实测）：unpack→bf16 写回→再读，HBM 流量
= int4 读 + bf16 写 + bf16 读 > 纯 bf16 读一遍 → decode 慢 5.5x。
B2 的解法：GEMM tile kernel 里 load uint8 → 位运算拆 nibble → ×scale →
tl.dot——**中间值不落 HBM**，权重读流量真实降为 1/3.9。

kernel 设计：
  grid = (cdiv(M, BM), cdiv(N, BN))
  每 program: x tile [BM, BK] bf16 原生读；W tile uint8 [BN, BK/2] 读；
    寄存器内: 拆 nibble → int4 → bf16 → 广播 group scale → [BN, BK]
    acc += tl.dot(x, w.T)
  约束（launch 前断言）: K % 64 == 0（BK=64 对齐 group=128 → 每 tile 恰
  2 个完整 group，scale 边界处理写死）、M 任意、N % BN == 0 可放宽（mask）。

数值纪律：acc fp32 累加（v4 kernel 同款——bf16 累加掉精度）。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_gemm_kernel(X, WPK, WS, Y, SCALE,
                         M, N, K,
                         stride_xm, stride_wn, stride_ym,
                         G: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Y = X @ dequant(W).T
    X: [M, K] bf16; WPK: [N, K/2] uint8; WS: [N, K/G] fp16(scale 已含在权重尺度);
    SCALE: 输出缩放（attention scale 等，通常 1.0）"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in range(0, K, BK):
        # ---- x tile [BM, BK] ----
        x_ptrs = X + rm[:, None].to(tl.int64) * stride_xm + (k0 + rk)[None, :]
        x = tl.load(x_ptrs, mask=(rm[:, None] < M), other=0.0)

        # ---- W tile: uint8 [BN, BK/2] → 两个连续半块 → dequant ----
        # 布局技巧：把 [BN, BK] 的偶列/奇列视为两个 [BN, BK/2] 的【并排加载】，
        # 各自独立 dequant 后 join——避免 interleave 重建的浪费。
        # 但数学上 Y = x @ w.T: 列序必须与 x 的列序一致（偶/奇交错）。
        # 换个等价做法: 把 x 也拆成偶/奇两半（[BM, BK/2] each），则
        #   Y = x_even @ w_even.T + x_odd @ w_odd.T   （纯代数，无重排！）
        rk2 = tl.arange(0, BK // 2)
        w_lo_ptrs = WPK + rn[:, None].to(tl.int64) * (K // 2) + k0 // 2 + rk2[None, :]
        wp = tl.load(w_lo_ptrs, mask=(rn[:, None] < N), other=0)      # uint8 [BN, BK/2]
        # 偶列（lo nibble）
        w_e = (wp & 0xF).to(tl.float32)
        w_e = tl.where(w_e >= 8.0, w_e - 16.0, w_e)
        # 奇列（hi nibble）
        w_o = (wp >> 4).to(tl.float32)
        w_o = tl.where(w_o >= 8.0, w_o - 16.0, w_o)

        # ---- group scale（BK | G 约束下每半块内的组索引连续算）----
        gs_e = (k0 + 2 * rk2) // G            # 偶列的全局 k
        gs_o = (k0 + 2 * rk2 + 1) // G        # 奇列的全局 k
        s_base = WS + rn[:, None].to(tl.int64) * (K // G)
        s_e = tl.load(s_base + gs_e[None, :], mask=(rn[:, None] < N), other=0.0).to(tl.float32)
        s_o = tl.load(s_base + gs_o[None, :], mask=(rn[:, None] < N), other=0.0).to(tl.float32)
        w_e, w_o = w_e * s_e, w_o * s_o

        # x 拆半（列 k → 偶列 k/2*2, 奇列 k/2*2+1）
        x_all = tl.load(X + rm[:, None].to(tl.int64) * stride_xm + (k0 + 2 * rk2)[None, :],
                        mask=(rm[:, None] < M), other=0.0)             # [BM, BK/2] 偶
        x_odd = tl.load(X + rm[:, None].to(tl.int64) * stride_xm + (k0 + 2 * rk2 + 1)[None, :],
                        mask=(rm[:, None] < M), other=0.0)             # [BM, BK/2] 奇

        we_t = tl.trans(w_e.to(tl.bfloat16))
        wo_t = tl.trans(w_o.to(tl.bfloat16))
        acc += tl.dot(x_all, we_t, input_precision="ieee")
        acc += tl.dot(x_odd, wo_t, input_precision="ieee")

    acc = acc * SCALE
    y_ptrs = Y + rm[:, None].to(tl.int64) * stride_ym + rn[None, :]
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def dequant_gemm(x: torch.Tensor, w_packed: torch.Tensor, w_scales: torch.Tensor,
                 group: int = 128) -> torch.Tensor:
    """x [M, K] bf16 × dequant(W) [N, K] → [M, N]。
    w_packed [N, K/2] uint8; w_scales [N, K/G] fp16。"""
    M, K = x.shape
    N = w_packed.shape[0]
    assert K % 2 == 0 and K % group == 0, f"K={K} 需 2|K 且 {group}|K"
    assert w_packed.shape[1] == K // 2
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BM, BN, BK = 16, 64, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _dequant_gemm_kernel[grid](
        x, w_packed, w_scales, y, 1.0,
        M, N, K, x.stride(0), w_packed.stride(0), y.stride(0),
        G=group, BM=BM, BN=BN, BK=BK, num_warps=4)
    return y


if __name__ == "__main__":
    # 数值对拍 + 性能对比（vs B1 的 unpack 路径和纯 bf16）
    import time
    from layers.quantized import rtn_quantize, dequantize

    torch.manual_seed(0)
    N, K = 2048, 2048
    w = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
    packed, scales = rtn_quantize(w.cpu(), 128)
    packed, scales = packed.cuda(), scales.cuda()

    for M, tag in [(8, "decode bs=8"), (4096, "prefill 4096")]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        # 数值参考: B1 反量化权重 matmul
        ref = (x.float() @ dequantize(packed, scales, 128).cuda().t())
        out = dequant_gemm(x, packed, scales, 128).float()
        diff = (out - ref).abs().max().item()
        rel = diff / ref.abs().max().item()
        print(f"M={M:<5}({tag:<13}) kernel vs B1 反量化 ref: max diff {diff:.2e} (rel {rel:.2e})")
        assert rel < 1e-2, "数值超出 bf16 舍入范围"

        # 性能三方对比
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
        print(f"          bf16 matmul {t_bf:.3f}ms | B2 kernel {t_k:.3f}ms "
              f"(x{t_k/t_bf:.2f} vs bf16)")
    print("\nALL PASS")
