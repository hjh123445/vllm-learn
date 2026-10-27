# v1：迭代级调度 + Orca 混合批次

把调度粒度从"请求"降到"迭代"：每个迭代之间完成的请求退出、新请求进来。
批的形状从 `[b, s]` 变成**平铺（flattened）**——prefill 和 decode 在同一个 forward 里跑。

## 快速开始

```bash
cd v1 && python example.py
```

## 代码结构

```
engine/
├── scheduler.py    迭代级调度：waiting/running 双队列 + cache slot 分配回收
├── model_runner.py step_mixed：唯一的平铺前向入口 + MixedAttnPlan（一次算好全层共享）
└── llm_engine.py   Engine：协调（每迭代 = 准入 → 合批 prefill + decode → 驱逐）
layers/attention.py KVCache（每请求一行）+ 平铺读写 + 混合分组注意力
layers/rotary_embedding.py  平铺版 RoPE（positions 是 [total_q]，无 b/s 维）
```

## 主要技术原理与实现

### 平铺批（flattened batch）

`[b, s]` 两个维度全部消失：所有 query token 摊平成一条 `[total_q, hidden]` 序列过模型。
**非注意力层（embedding/norm/MLP/投影/lm_head）对 total_q 无差别，零改动天然兼容**；
只有 attention 需要"批"的概念。

### Orca 混合批次：prefill 和 decode 同一个 forward

新准入请求的 prefill token 和在途请求的 decode token 拼在同一条平铺序列里。
attention 按 plan 分两组算（`layers/attention.py`）：

- **decode 组**：全部 s=1 的请求天然同形 → 合批成 `[n, h, 1, L]` 一条批注意力（热路径，
  每步都跑）；
- **prefill 行**：每行 s 各不相同没法合批 → 逐行 `[1, h, s_i, L_i]`（每请求只发生一次，
  不在热路径）。

plan（`engine/model_runner.py::step_mixed`）在模型外**一次算好、全部 28 层共享**：
每 token 的 cache 行号、写位置、mask、每序列的 flat 切片。lm_head 只在要出 token 的位置算。

### KVCache：每请求一行 + 平铺读写（`layers/attention.py`）

`[max_requests, kvh, max_len, hd]`，每请求独占一行，**cache 槽位 = 请求自己的真实位置**。
- 写：`write_flat` 按 token 的线性下标**一次散射**（零 python 循环零 `int()` 设备同步——
  v0 的逐行循环实测 472µs/层，这里 110µs）；
- 读：decode 组 gather，单行是 view 零拷贝。

### 迭代级调度（`engine/scheduler.py`）

`admit()`（有 slot 就准入）→ prefill → `decode_step(全体 running)` → `evict()`（完成即回收 slot）。
新请求在**迭代边界**被发现并融入——这就是在线服务的最小内核。

## 与 v0 的差异

| v0 | v1 |
|---|---|
| batch 跑完整批，slot 空转 | 每迭代准入/驱逐，气泡 35% → 0 |
| 左 padding + 共享 cache 形状 | 无 padding，变长靠 mask |
| `[b, s]` 张量 | 平铺 `[total_q, hidden]` |
| prefill/decode 分开 | 同一个 forward 混跑 |

## 正确性与性能

- 与 v0 单请求路径 **bf16 逐 token 等价**（accuracy T3）；fp32 HF 100% 对齐
- TTFT avg 717→**v1 混合批次后** 吞吐 146.8 tok/s（0.6B）；TPOT 39.5ms

## 已知限制（有意的）

- slot 是不可分割的分配单位：一条 50 token 请求占一整个 448MB slot（v2 的 paged 修这个）
- 显存仍按 batch_size × max_seq_len 预分配，无预算无抢占
