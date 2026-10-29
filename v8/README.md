# v8：上下文并行 CP / DCP

建立在 **v7**（投机解码）之上——版本叠加，目录 = v7 全部 + CP/DCP 组件。
前置依赖：v6.4 调度集中化（DCP 块染色要求所有 rank 对 block 表有一致视图）。

## 小版本演进

| 小版本 | 主题 | 状态 | 独立文档 |
|---|---|---|---|
| v8.0 | LSE 合并数学（partial + log-sum-exp 加权合并） | ✓ ALL PASS | 本文 §v8.0 |
| v8.1 | DCP2 双卡端到端（块染色 + partial/LSE + ag 合并） | ✓ ALL PASS | 本文 §v8.1 |
| v8.2 | 混合并行坐标系统接线引擎（TP2×DCP2 四卡） | ✓ 78/78 一致 | 本文 §v8.2 |
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

## v8.3-v8.4 规划

- **v8.3**：CP2 prefill——长序列对半分，各卡 partial attention + LSE 合并
  （复用 v8.0 数学），验证 64K 序列的 TTFT 加速；
- **v8.4**：ag_rs → a2a 后端（AllGather/ReduceScatter 换 All-to-All，
  3 次 NCCL→2 次/层）+ triton 化 partial attention（v8.2 的 eager 实现慢 8 倍）
  + 128K 压测。

## v8.2 实现记录：混合并行接线引擎（TP2×DCP2 四卡）

**架构**（`tests/run_dcp_tp.py` 三相位验证：单卡 base / TP2 / TP2×DCP2）：

```
rank = tp*cp_size + cp   （ParallelCoords row-major: (dp,tp,cp,ep,pp)）
r0=(tp0,cp0) r1=(tp0,cp1) r2=(tp1,cp0) r3=(tp1,cp1)

tp 组 = 坐标除 tp 外全同 = {r0,r2} {r1,r3}   ← o_proj/MoE 后的 allreduce（头切分）
cp 组 = 坐标除 cp 外全同 = {r0,r1} {r2,r3}   ← DCP partial/LSE 合并（位置切分）
```

**TP 与 DCP 的正交性**在组结构上直接可见：TP 组内各 rank 头不同、cache 分片相同
（allreduce 求和）；cp 组内各 rank 头相同、分片不同（LSE 加权合并）。
同一个全局块表，两种切法互不干扰——这就是多维坐标向量设计的意义。

**三层接线**：
1. **store 门控**（`kernels.py`）：全局块 g 归 `g%cp_size==cp_rank` 的 rank，
   本地槽位 = `g//cp_size`（静态映射保证增量写落在同一物理位置）；
   非本 rank 颜色的 token 槽位置 -1，融合 kernel 的 store 带 mask 跳过；
2. **partial attention**（`attention.py _dcp_attention`）：只对本 rank 颜色的行
   算未归一化 partial + m + l（causal 门 = 块表展开位置 ≤ q 位置）；
3. **合并**：part/m/l pack 成单 buffer **1 次** all_gather（cp 组）→ 本地
   `out = Σ e_i·part_i / Σ e_i·l_i`（约定 B：part 未归一化，w=exp(m_i-m*)）。

**验证**（Qwen3-Omni-30B，48 tok greedy ×3 请求）：

| 配置 | wall | 输出 vs 单卡 base |
|---|---|---|
| base（单卡） | 5.0s | — |
| TP2 | 6.8s | **ALL MATCH 78/78** |
| TP2×DCP2（4 卡） | 42.0s* | **ALL MATCH 78/78** |
| DCP 本地池 | 575→288 blocks | **KV 显存精确减半**（3.62GB/rank） |

*DCP 慢 8 倍是预期内的：partial attention 是纯 PyTorch 逐层实现（`_dcp_attention`
的 python 块表展开循环 + eager einsum），triton 化在 v8.4（decode 路径复用
v4 的 paged flash kernel + 每 rank 只扫自己颜色的行）。本版的裁决点是
**正确性与正交性**：混合坐标下 4 卡输出逐 token 一致 + 显存减半。

**踩坑实录**（两个，都是硬骨头）：

1. **NCCL 多 communicator bootstrap 竞态（25 分钟死锁）**：`new_group` 创建的
   tp/cp 组第一次使用时惰性 wire-up，多组交错初始化时 TCP bootstrap 连接被拒
   （`socketPollConnect: Connection refused` 重试 35 次）——4 卡下稳定复现
   于 layer0 的第一个 cp 组 all_gather。**根因**：`init_process_group` 未绑
   `device_id`，NCCL communicator 的设备亲和在多组并发建立时竞态。
   **修复**：`init_process_group("nccl", device_id=...)` + `warmup_groups`
   （每维组启动时做一次小 all_reduce 强制完成 wire-up）。定位工具是
   faulthandler.dump_traceback_later——把"挂在哪个 collective"变成铁证
   （两个进程在 tp_allreduce、两个在 cp all_gather 的交叉等待）。
2. **LSE 合并的约定混用**：partial 输出是未归一化的（Σexp(s-m_i)·v），合并却
   用了 v8.1 归一化约定的公式（w 含 l_i）——两套自洽约定 (A) parts 已归一化 +
   w=exp·l；(B) parts 未归一化 + 分母 Σ exp·l——实现里 A 的公式配 B 的 parts，
   输出乱码（从 tok0 就错）。单进程数学对拍（不经 NCCL，合成 KV）3 分钟定位。

**正交性收益**：4 卡 = TP2 的计算吞吐 × DCP2 的 cache 容量——同一批 rank 上
同时拿到两个维度的扩展。DCP 不扩 world size 的意义在此：128K 单请求场景
TP8 已用满 8 卡时，DCP 是继续扩 cache 的唯一手段（v8.4 压测验证）。

## 与投机解码（v7）的交互

v8 建立在 v7 之上：投机 verify 的多 token 一次前向与 DCP 分片天然互补——
verify 批次扩大 k 倍摊薄 DCP 的每步通信，DCP 让长序列的 verify cache
显存减半。两者组合是长序列 + 低时延场景的方向（v8.4 压测验证）。
