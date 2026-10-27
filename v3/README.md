# v3：nano-vllm 调度栈 + 我们的模型

把 nano-vllm 的调度栈**原样搬入**——APC 前缀缓存、block 引用计数、chunked prefill、
max_num_batched_tokens、抢占——模型沿用我们的 v2 实现。ModelRunner / LLMEngine 重写为单卡单进程。

## 快速开始

```bash
cd v3 && python example.py
```

## 代码结构

```
# ↓ 原样搬入 nano-vllm（仅改 import 路径）
engine/block_manager.py   Block 引用计数 + xxhash 链式哈希的前缀缓存（APC）
engine/scheduler.py       prefill 优先 + chunked prefill + max_num_batched_tokens + 抢占
engine/sequence.py        Sequence（token_ids 全量保存 + block_table + 分块语义）
utils/context.py          Context 全局单例（cu_seqlens/slot_mapping/context_lens/block_tables）
# ↓ 我们的实现
engine/model_runner.py    重写：warmup → nano 公式算池子 → 注入每层 → prepare 构造 Context
engine/llm_engine.py     重写：单卡单进程（去掉 mp/共享内存/分布式）
layers/ models/           v2 的实现（平铺 + context 版 attention）
```

## 主要技术原理与实现

### APC：前缀缓存 + 引用计数（`engine/block_manager.py`）

每个 block 的内容做**链式哈希**（`hash = xxhash(前缀 hash + 本块 token_ids)`）登记在
`hash_to_block_id` 表里。新请求 prefill 时逐块查哈希，命中就把现成 block 接进自己的
block table（`ref_count += 1`），**跳过这些 token 的计算**——同 system prompt 的多轮对话直接受益。
驱逐时 `ref_count` 归零才真正回收。写 cache 是"边前向边写"，所以命中前缀意味着
prefill 输入从命中长度开始切。

### chunked prefill（`engine/scheduler.py`）

`max_num_batched_tokens`（默认 16384）限制**每次迭代进入 forward 的总 token 数**。
长 prompt 切成块分多轮混入 decode 流——没有它，一个 4000-token 的 prefill 会把同批所有
在途请求那一步的 TPOT 从 40ms 打到几百 ms。这是我们欠的最后一课（v2 的 README 里记着）。

### Context（`utils/context.py`）

nano 的约定：`set_context(is_prefill, cu_seqlens_q/k, max_seqlen_q/k, slot_mapping, context_lens, block_tables)`。
- prefill：`cu_seqlens_q/k` 是平铺批里每序列的边界（**绝对前缀和**，键长必须相减得到！），
  `slot_mapping` 是每 token 的全局槽位；
- decode：`context_lens`（每序列已写长度）+ `block_tables`（padded）。

### 我们的 attention 读 Context（`layers/attention.py`）

prefill 逐序列按 cu_seqlens 切分，causal **底右对齐**（chunk 是 k 的尾段，支持前缀缓存）；
mask 用**全局加性 causal 表的切片**（视图零拷贝——每层每序列重建 mask 会让 prefill 2.02s → 0.16s，
实测踩过）。decode 组合批：`context_lens` 做长度 mask。

### 池子：nano 的公式（`engine/model_runner.py::allocate_kv_cache`）

`num_blocks = (总显存×利用率 − 已用(含其他进程) − warmup峰值 + 当前) ÷ 每 block 字节`，
block_size=256。warmup 用临时池 + 手动指块（我们的 attention 要从 cache 读，
不像 nano 的 flash prefill 能无 cache 直算）。

## 与 v2 的差异

| v2 | v3 |
|---|---|
| block 16 tok，自管账本 | block 256 tok，nano 的 BlockManager（APC + ref_count） |
| 无 APC / chunked prefill | 两者都有（原样搬入） |
| 自写 Context | nano 的 Context 约定（cu_seqlens 体系） |
| 逐请求温度采样（v3 顺手修掉 nano 的 greedy 禁止 assert） | |

## 正确性与性能

- fp32 HF 逐 token 100% 对齐（含 chunked prefill / 前缀缓存路径）
- 0.6B：TPOT 37.6ms / 156.4 tok/s，TTFT avg 117ms（prefill 单 kernel 化后 0.16s/批）

## 已知限制（有意的）

- attention 是朴素 PyTorch（gather + matmul + softmax），CPU 发射瓶颈（v4 手写 kernel）
- max_num_seqs 只限每步批大小，**不限总并发**（吞吐优先策略，与 vLLM 的时延优先策略不同，
  压测口径对比见 benchmark/PERFORMANCE_NOTES §9）
