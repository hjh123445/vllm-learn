# v8：上下文并行 CP / DCP

建立在 **v7**（投机解码）之上——版本叠加，目录 = v7 全部 + CP/DCP 组件。
前置依赖：v6.4 调度集中化（DCP 块染色要求所有 rank 对 block 表有一致视图）。

## 小版本演进

| 小版本 | 主题 | 状态 | 独立文档 |
|---|---|---|---|
| v8.0 | LSE 合并数学（partial + log-sum-exp 加权合并） | ✓ ALL PASS | 本文 §v8.0 |
| v8.1 | DCP2 双卡端到端（块染色 + partial/LSE + ag 合并） | ✓ ALL PASS | 本文 §v8.1 |
| v8.2 | 混合并行坐标系统接线引擎（TP×DCP 组合） | 待做 | — |
| v8.3 | CP2 prefill（长序列对半分 + partial attention） | 待做 | — |
| v8.4 | a2a 通信后端（3 次 NCCL→2 次）+ 长序列压测 | 待做 | — |

## CP/DCP 是什么（一句话版）

- **CP（prefill 计算并行）**：序列维度切到多卡解决 O(L²) 计算放不下（扩 world size）；
- **DCP（decode cache 并行）**：KV cache 沿序列交错分片解决 O(L) 显存放不下
  （**不扩 world size**——复用 TP rank）；
- 两者正交；**DCP ≠ DP**：DP 切请求（每卡完整跑一个请求），DCP 切单请求的
  cache（多卡协作跑一个请求）——128K 单请求场景 DP 无能为力，DCP 是唯一解。

完整原理（vLLM 0.31 源码调研）：[DESIGN.md](DESIGN.md)。

## v8.0 —— LSE 合并数学 ✓

`lse_merge_test.py`：CP/DCP 共用的核心数学——各卡的 partial attention 带
softmax 的 m（最大值）和 l（分母），合并端 `w_i ∝ exp(m_i - m*)·l_i` 加权求和。

**验证**：N=1/2/3/4/8 分片 vs 全量 attention，fp32 **逐位一致**（diff ~3e-7）。
这是在线 softmax 数学在空间分片上的扩展（与 v4 decode kernel 的流式在线
softmax 同一数学：时间流式块 → 空间卡间分片）。

## v8.1 —— DCP2 双卡端到端 ✓

**三层实现**：
1. `parallel_coords.py`：**混合并行的地基**——进程多维坐标向量
   `(dp, tp, cp, ep, pp)`，rank = row-major 线性化解码，每维一个进程组
   （对齐 vLLM parallel_state 设计）。单元验证：坐标往返编解码 8/8 + 分组成员正确；
2. `layers/dcp.py`：DCPCacheCoordinator（块染色：`block_id % dcp_size ==
   cp_rank` 归本卡）+ partial_attention_with_lse + lse_merge_stack；
3. `test_dcp.py`：双卡各持一半 KV，decode 步每卡算自己分片的 partial+LSE →
   all_gather → 本地 LSE 合并。

**验证**：8 步 decode 的 DCP 合并 vs 全量 attention **max diff ~1e-8 ALL PASS**；
KV cache 显存精确减半（32/64 位置，4/8 块）。

**为什么交错染色而不是各持一半**：causal attention 下靠后位置 attends 更多，
连续分片把长位置负载全压给后段 rank；交错让每卡工作量均匀（vLLM 的
`interleave_size` 同理）。

## v8.2-v8.4 规划

- **v8.2**：坐标系统接进 v6 引擎（parallel_worker 改用 ParallelCoords），
  跑通 TP2×DCP2 四卡组合（TP 组内 allreduce + cp 组内 ag/merge）；
- **v8.3**：CP2 prefill——长序列对半分，各卡 partial attention + LSE 合并
  （复用 v8.0 数学），验证 64K 序列的 TTFT 加速；
- **v8.4**：ag_rs → a2a 后端（AllGather/ReduceScatter 换 All-to-All，
  3 次 NCCL→2 次/层）+ triton 化 LSE 合并 kernel + 128K 压测。

## 与投机解码（v7）的交互

v8 建立在 v7 之上：投机 verify 的多 token 一次前向与 DCP 分片天然互补——
verify 批次扩大 k 倍摊薄 DCP 的每步通信，DCP 让长序列的 verify cache
显存减半。两者组合是长序列 + 低时延场景的方向（v8.4 压测验证）。
