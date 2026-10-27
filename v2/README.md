# v2：PagedAttention + 显存统一管理

KV cache 按 16-token block 切块、每请求一张 block table，池子大小由 warmup **量出来**，
准入/驱逐/抢占全按 block 账本走。更多设计推演见 `DESIGN.md`。

## 快速开始

```bash
cd v2 && python example.py
```

## 代码结构

```
engine/
├── kv_cache.py     KVCacheManager：唯一的显存所有者（paged 池 + block table + 账本）
├── context.py      Context 全局单例：attention 上下文（slot_mapping/分组/mask）
├── scheduler.py    schedule()→(reqs, is_prefill) + update()：准入/驱逐/抢占全在内部
├── request.py      Request：单请求状态（独立文件）
├── model_runner.py step(reqs, is_prefill)：平铺前向 + set_context + warmup 显存 profile
└── llm_engine.py   Engine：主循环只剩 7 行
```

## 主要技术原理与实现

### Paged KV cache（`engine/kv_cache.py`）

池子一片 `[layers, blocks, block_size, kvh, hd]`（block=16 tok，1.84MB/block），
每请求一张 block table（逻辑连续物理离散）。**全局槽位 = block_id × block_size + 块内偏移**，
可平铺索引 → 写入 cache 只需一次 index_put（`write_slot`）。
读分两条：`read_batch`（decode 组整组一次 gather）/ `read_seq`（prefill 行，单块是 view）。

### 池子大小：warmup 量出来，不是算出来（`engine/model_runner.py::_warmup_profile`）

```
1. 载权重 → memory_allocated() 量出权重实占
2. 建临时小池，跑 dummy 最大压力前向（prefill + decode 两种形状都量）
   → max_memory_allocated() 量出激活峰值
3. max_blocks = (总显存 × gpu_memory_utilization − 权重 − 激活峰值) ÷ 每 block 字节
```

### 准入、增长与抢占

```
准入: free_blocks ≥ ceil(prompt_len/16) + 1   （+1 是水位储备）
增长: decode 写位置 pos 跨 block 边界(pos % 16 == 0)时领新 block
抢占: 领不到 → 打回占块最多的请求(reset + 回 waiting 队头)，之后从头重算(recompute preemption)
```

牺牲对象选"块最多"的——它释放的显存最多，最不容易再触发抢占（正反馈抑制）。

### 全局 Context（`engine/context.py`）

模型前向签名只剩 `(input_ids, positions)`：slot 映射、混合分组、mask 全在 Context 里，
`ModelRunner.step` 每步 `set_context` 一次，每层 attention `get_context` 读。
平铺批按形状分组算：decode 组合批（热路径）+ prefill 逐行。

### attention 分组（`layers/attention.py`）

decode 组 s=1 天然同形合批（一次 gather + 一条 `[n,h,1,L]` 批注意力）；
prefill 行 s 各异逐行。实测朴素逐序列（每层每序列一次注意力）TPOT 会到 100ms，
**热路径必须组批**——这是"长度不同"的正确处理：同形合批 + 异形单独 + mask，不是 padding。

## 与 v1 的差异

| v1 | v2 |
|---|---|
| slot = 不可分割的 448MB | block = 16 tok 1.84MB，按实际用量占 |
| 池子 = batch_size × max_seq_len 拍脑袋 | warmup 量出权重+激活 → 算出 block 数 |
| 无预算无抢占 | block 准入 + 水位 + recompute 抢占 |
| plan 传参 | Context 全局单例 |

## 正确性与性能

- fp32 HF 逐 token 100% 对齐；v0/v1/v2 单请求路径逐 token 等价（accuracy 全套）
- 0.6B：TPOT 36.6ms / 141.5 tok/s；8B 兼容（GQA 4:1 / untied 零改动跑通）
- 同显存下 1024-token 请求并发 16 → 203 条（`benchmark/capacity.py`）

## 已知限制（有意的）

- 无引用计数 / 前缀复用（APC）、无 chunked prefill / max_num_batched_tokens（v3 搬 nano 全套）
- block table 每步 CPU 构造 + HtoD；采样整批共用第一个请求的参数
