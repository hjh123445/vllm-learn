# v8 设计：上下文并行（CP）与解码上下文并行（DCP）

> 状态：设计定稿。前置依赖：v6.4 调度集中化（三个 open issue 的共同解）。
> 本文档基于 vLLM 0.31 源码调研（`vllm/config/parallel.py`、
> `vllm_flash_attn/flash_attn_interface.py`）+ 我们的 KV cache 架构。

---

## 0. CP/DCP 是什么，解决什么问题

**长序列推理的两个独立瓶颈**：

1. **prefill 计算量 ∝ L²**：128K 序列的 attention 是 O(L²)，单卡算力不够 → **CP（Context Parallel，vLLM 里叫 PCP）把序列维度切到多卡**；
2. **decode 的 KV cache 显存 ∝ L**：128K 序列 × 多请求的 KV cache 撑爆显存 → **DCP（Decode Context Parallel）把 KV cache 按序列维度分片到多卡**。

两者正交：PCP 切**计算**（prefill 时），DCP 切**显存**（decode 时）。vLLM 0.31 的配置注释原文：

```
PCP: "Number of ranks that split prefill sequence computation. PCP expands
      the process world size."
DCP: "Number of ranks that shard the decode KV cache. DCP does not expand
      the process world size. Without PCP, DCP reuses TP ranks."
```

**关键区别**：PCP 扩 world size（多进程），DCP **不扩**——它复用 TP 组的 rank，
让同一个模型的 TP rank 顺便分摊 KV cache。这就是为什么 DCP 特别优雅。

## 1. CP 的工作原理（prefill 序列切分）

### 1.1 切分方式

序列 L 分成 N 段，rank i 持第 i 段的 q/k/v：

```
序列: [-------- L = 8192 --------]
CP2:   rank0: [段0: 0..4095]      rank1: [段1: 4096..8191]
```

### 1.2 核心难点：attention 的跨段依赖

段 i 的 query 要 attend 段 j 的 key（跨 rank）。两种解法：

**方案 A（vLLM 的 flash CP kernel，我们采用的）**：
每对 (i,j) rank 交换 q 或 k/v 一次，各自算 partial attention，再用
**LSE（log-sum-exp）加权合并**：

```
partial_attn(i→j) = softmax(q_i k_j^T / √d) v_j        # rank i 用对方的 k_j v_j
合并: out_i = Σ_j w_j · partial_j, 其中 w_j ∝ exp(LSE_j - max(LSE))
```

LSE 合并是在线 softmax 的跨卡扩展（与 v4 的 decode kernel 同一数学——
只是把"时间上的流式"变成"空间上的分片"）。

**方案 B（Ring Attention）**：k/v 沿 ring 轮转 L/N 轮，每轮算 partial。
通信量更大但无需成对交换。

### 1.3 CP 的因果掩码问题

段 i 的 query 只 attend ≤ 它的 key：右段的 rank 收到的 partial 需要跳过
"未来"部分（vLLM 的 `cp_rank` 参数传给 flash kernel 处理 causal 边界）。
我们的 prefill kernel（`_prefill_attn_kernel`）已有底右对齐的 causal 处理，
扩展点是接收 `remote_k/v` 指针。

## 2. DCP 的工作原理（decode KV cache 分片）

### 2.1 切分方式（interleave，不是连续块）

vLLM 的 `dcp_kv_cache_interleave_size`：KV cache 沿序列**交错分片**：

```
序列位置:  0  1  2  3  4  5  6  7 ...
DCP2:     r0 r1 r0 r1 r0 r1 r0 r1   （interleave=1）
```

为什么交错而不是各持一半：**负载均衡**——decode 的 causal attention 里，
靠后的位置 attends 更多（O(L) vs O(1)），连续切分会把长位置的负载全压给
持后段的 rank。交错让每卡的 attention 工作量均匀。

### 2.2 decode 一步的 DCP 流程

```
输入: x [b, h]（新 token 的 hidden，各 rank 相同——TP 已同步）
1. 每个 rank 对自己的 KV 分片算 partial attention:
   partial_i = softmax(q · K_i^T) V_i          # K_i 是本 rank 持的交错位置
   LSE_i = log(Σ exp(q · K_i^T))               # 本地 softmax 的归一化因子
2. 通信合并（vLLM 的两种后端）:
   ag_rs:  AllGather(partial, LSE) → 每 rank 全量 → 本地加权合并 → ReduceScatter(out)
           3 次 NCCL 调用/层
   a2a:    All-to-All(partial 输出 + LSE) → Triton kernel 合并
           2 次 NCCL 调用/层（vLLM 注释: "Reduces NCCL calls from 3 to 2 per
           layer for MLA models"）
3. out = Σ_i softmax(LSE_i - max) · partial_i   # LSE 加权合并
```

### 2.3 DCP 的显存收益

KV cache / N（N=DCP size）。对 128K 长序列：单卡 128K × 115KB/token ≈ 14GB/请求
→ DCP4 后 3.5GB/请求 → **同样显存 4 倍并发**。这是 DCP 的核心卖点：
**它切的是"每请求的 cache"，与 batch 维度的 DP 正交**。

## 3. 我们的实现方案（v8）

### 3.1 与现有架构的对接点

| 组件 | 现状 | CP/DCP 改造 |
|---|---|---|
| `prefill_attn` kernel | 本地 block 表 causal | **+CP**: 接收 remote k/v 指针，算 partial + LSE，LSE 合并 |
| `decode_attn` kernel | 本地 block 表在线 softmax | **+DCP**: 每卡只算自己的交错分片 → partial + LSE 输出 |
| KV cache 布局 | `[blocks, bs, kvh, hd]` 按请求 | **+DCP**: block 分配时按 rank 交错染色（block i 归 rank i%N） |
| Context | slot_mapping 全局 | **+DCP**: slot_mapping 只映射本 rank 的块 |
| scheduler | 全请求全 cache 视角 | 不变（DCP 对 scheduler 透明！只改 cache 分配和 attention） |

**DCP 优雅之处**：scheduler 完全不用改——它只看逻辑 block 表，
物理上哪些块在哪张卡由 cache 分配器决定。这正是我们 v2 就把
`KVCacheManager` 做成显存唯一所有者的回报。

### 3.2 实现路线（每步可验证）

| 步 | 内容 | 验证 |
|---|---|---|
| 8.0 | **DCP 的 LSE 合并数学**（纯 PyTorch 单元测试）：partial+LSE 加权合并 vs 全量 attention 逐位对拍 | fp32 diff = 0 |
| 8.1 | **DCP2 双卡**：cache 块交错染色 + decode kernel 改造（每卡算自己分片的 partial/LSE）+ ag_rs 合并 | 与 DCP1 逐 token 等价 |
| 8.2 | **DCP 的 ag_rs → a2a 优化**：AllGather/ReduceScatter 换 All-to-All + triton 合并 kernel，通信 3 次→2 次/层 | 通信量微基准 + TPOT 对比 |
| 8.3 | **CP2 prefill**：长序列对半分，partial attention + LSE 合并（复用 8.0 的数学） | 与 CP1 逐 token 等价 |
| 8.4 | **长序列压测**：64K/128K 序列的 TTFT（CP）+ 并发数（DCP）对比 DCP1/CP1 | 吞吐/显存/TTFT 三维 |

### 8.3 的实验设计（与 EXPERIMENT_PLAN 的格式一致）

```
自变量: 序列长度 ∈ {8K, 32K, 64K, 128K} × CP ∈ {1, 2, 4}
固定: 模型、batch=1（长序列场景 batch 天然小）
测量: prefill 时延（CP 的加速比应随 L² 计算的分摊线性↑）
预期: CP2@128K ≈ 2× CP1@128K（计算分摊），CP 的通信量 ∝ L/N（可摊薄）

DCP 实验另一维:
自变量: 并发请求数（显存约束下的最大值）
测量: DCP1 vs DCP4 的最大并发（应 4×）+ TPOT 变化（DCP 的每步通信开销）
```

## 4. CP/DCP 与其他并行的组合

| 组合 | 语义 | 适用 |
|---|---|---|
| DCP×TP | DCP 复用 TP rank（vLLM 默认）：TP 组内顺便分 cache | 标准配置 |
| CP×EP | prefill 计算分 CP，MoE 分 EP | DeepSeek 式长序列 MoE |
| DCP×DP | cache 分片（单请求显存）× 请求副本（总吞吐） | 长序列在线服务 |

**DCP 与 DP 的区别（易混）**：DP 切**请求**（每卡独立跑完整请求，cache 完整）；
DCP 切**单请求的 cache**（多卡协作跑一个请求）。128K 单请求场景 DP 无能为力
（一张卡装不下它的 cache），DCP 是唯一解。

## 5. 与 vLLM 的对照

| | vLLM 0.31 | 我们 v8 |
|---|---|---|
| PCP | `--prefill-context-parallel-size`（扩 world） | 8.3 |
| DCP | `--decode-context-parallel-size`（复用 TP rank） | 8.1 |
| DCP 通信 | ag_rs / a2a 两种后端 | 8.1 用 ag_rs，8.2 实现 a2a |
| CP kernel | vllm_flash_attn 的 cp_rank 参数 | 扩展我们的 triton kernel |
| interleave | `cp_kv_cache_interleave_size` | 8.1 的块染色 |

## 6. 前置依赖

v6.4 的调度集中化（rank0 决策 + broadcast）——DCP 的 cache 分片要求
所有 rank 对 block 表有一致视图（谁持哪块），这正是集中化调度提供的保证。
