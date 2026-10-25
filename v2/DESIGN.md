# v2 设计文档：PagedAttention 与显存管理

v1 → v2 的全部优化点、实现细节、设计考量。数据全部实测（复现方法见 `benchmark/`）。

---

## 1. v1 的显存分配逻辑是什么？（v2 要解决的问题）

v1 **没有任何显存管理逻辑**：cache 在构造函数里 `register_buffer` 满额预分配
`[batch_size, kv头, max_seq_len, dim]`，拍脑袋的两个参数（batch_size × max_seq_len）
就是显存预算。没有 warmup、没有激活峰值测量、没有水位、没有 OOM 防护：

| 问题 | v1 现状 | 后果 |
|---|---|---|
| 池子多大 | batch_size × max_seq_len 拍脑袋 | 显存小了浪费（实测利用率 7%），大了启动即 OOM |
| 有没有 warmup profile | 没有 | 激活峰值是未知的，预算只能瞎留 |
| 准入判据 | `len(free_slots)` 个数 | 对显存盲：短请求也值一整个 slot |
| 请求增长 | 无检查 | 写越界只会炸 assert |
| OOM | 无防护 | 超了就崩，没有抢占/回退 |

## 2. v1 → v2 全部优化点清单

按依赖顺序（每条都有实测依据，出处标在括号里）：

| # | 优化点 | 原理 | 实测收益 |
|---|---|---|---|
| 1 | **RmsNorm eps + 原地操作修复** | bf16 下 `.float()` 拷贝掩盖了 in-place bug；fp32 下 residual 被毁 | fp32 与 HF 从 0% → 100% 逐 token 对齐（accuracy T4） |
| 2 | **cache 写向量化** | 逐行 python 循环 = 每 layer 每步 16 次 kernel 发射 + `int()` 每行一次设备同步 | 472→110 µs/层（微基准），TPOT −8%（PERFORMANCE_NOTES §3.2） |
| 3 | **cache 读改 block gather** | 行 gather 拷贝整行（含尾部没用到的空间），block gather 只搬需要的 block | 36.2→26.8 µs/层（微基准） |
| 4 | **池子改全局一片 `[layers, blocks, kvh, 16, dim]`** | block 分配跨层原子（`pool[:, 37]` 一个下标）；显存账本唯一 | 使 #5/#6/#7 成为可能 |
| 5 | **warmup 显存 profile** | 激活峰值只能量不能算（kernel 的临时 workspace、分配器对齐都在纸面模型外） | 池子大小从拍脑袋变成"量出权重 + 量出激活 → 算出 block 数" |
| 6 | **准入按 block 预算** | `can_allocate(prompt_len)`：显存账本驱动准入 | TTFT p99 304→47 ms（stress） |
| 7 | **抢占（recompute preemption）** | block 耗尽时不崩：把占块最多的请求打回 waiting 重算 | 尾长流不会炸引擎 |
| 8 | **读 mask 向量化构造** | v1 的 `valid[:, None, None]` 在 [b,1] 上广播出 5 维 mask | 修复一个真实 shape bug（见 §4 踩坑） |

## 3. v2 显存管理：实现细节

### 3.1 池子的物理布局

```
k/v: [num_layers=28, num_blocks, num_kv_heads=8, block_size=16, head_dim=128]
      ↑ 每层一个切片（attention 拿 view，零拷贝）
                      ↑ block 粒度管理（分配/回收/抢占的单位）
```

- 一个 block = 1.84 MB（全部层 × k/v × 8 头 × 16 tok × 128 dim × 2B）
- 每请求一张 block table：`tables[req_id] = [block_idx, ...]`，逻辑顺序 = token 顺序，
  物理块离散——和 OS 页表同构，slot→block 的映射 freed 后立即可复用
- 碎片只有每请求最后一个不满 block（平均浪费 8 tok），v1 的内部碎片实测 93%

### 3.2 池子大小怎么定（warmup profile，量出来的三步）

```
1. 载权重 → memory_allocated() 量出权重实占          (1.19 GB)
2. 建临时小池（256 block），跑 dummy 最大压力前向：
   prefill(batch=8, 256 tok) + decode_step —— 覆盖两种激活形状
   → max_memory_allocated() 量出峰值，减掉临时池   (激活 0.03 GB)
3. max_blocks = (总显存 × gpu_memory_utilization − 权重 − 激活) ÷ 每 block 字节
   → rebuild(max_blocks)，临时池释放
```

注意 dummy 覆盖 prefill 和 decode 两种形状——prefill 的激活远大于 decode
（[8, 256, hidden] 全序列中间量），只测 decode 会低估峰值。这就是 vLLM
`determine_available_memory` 的同款流程。

### 3.3 准入、增长与抢占

```
准入:  free_blocks ≥ ceil(prompt_len/16) + 1     (+1 是水位储备)
增长:  decode 写位置 pos 跨 block 边界时 (pos % 16 == 0) 领新 block
抢占:  领不到 → 打回占块最多的请求（reset + 回 waiting 队头），之后从头重算
```

为什么是 recompute 而不是 swap 到 CPU：v2 简化（swap 要维护 CPU 副本 + 回填），
vLLM 两种都有。抢占代价 = 请求已生成部分作废，所以牺牲对象选"块最多"的——
它占的显存最多，释放后最不容易再触发抢占（正反馈抑制）。

### 3.4 attention 的 paged 读写（全部向量化）

- **写**：每个目标元素的 (block 号, block 内偏移) 算成线性下标张量，一次 scatter 写完
  整个 batch 的所有行——零 python 循环、零 `int()` 同步。
- **读**：`pool[layer][tabs]` gather 成 `[b, nblk, kvh, 16, dim]`，permute+reshape 成
  连续 `[b, kvh, nblk*16, dim]` 再截到 read_len。block 内尾部多出的槽位由 mask 挡住
  （mask 本来就要构造，没有额外代价）。
- attention 本体与 v1 相同（GQA repeat + softmax + mask）——**paged 的复杂度全部
  被关在 KVCacheManager 的 read/write 两个函数里**，模型层无感知。

### 3.5 引擎循环（与 v1 相同 + 抢占一步）

```
while scheduler.busy:
    admit()（block 预算）→ 逐条 prefill → 采样 → evict
    for req in running: while not ensure_append(req): preempt(pick_victim())
    decode_step(全部活跃请求合批) → 逐请求 step → evict
```

## 4. 实现过程中的踩坑（值得记住的）

1. **`tabs[:, positions // bs]` 的多维高级索引会插维**：索出 [b, 1, s] 而非 [b, s]，
   后续广播全错。要用 `tabs[arange(b)[:, None], positions // bs]`。
2. **`valid[:, None, None]` 作用在 [b,1] 张量上**：本意 [b, 1, 1, L]，实际 [b, 1, 1, 1, L]，
   5 维 mask 和 scores [b, h, 1, L] 相加广播成 [b, h, 1, 1, L]，softmax 之后维度
   错位——报错在 o_proj（隔了两层），靠逐模块打印 shape 才定位。教训：**对 [b,1]
   张量加维度前先 flatten 成 [b]**。
3. **`block_bytes` 切错维度**：`k[0]` 是 [blocks, ...]（整层的全部 block），
   一个 block 应该是 `k[:, 0]`（全部层的同一个 block 号）。量纲检查救了这个 bug：
   294 MB/block 显然不对（1 token 才 112 KB）。

## 5. 与 vLLM 的差异（诚实的清单）

| | 本 v2 | vLLM |
|---|---|---|
| block table | 每步在 CPU 上构造 + HtoD 拷贝 | 常驻 GPU，增量更新 |
| paged attention kernel | gather 成连续 buffer + 标准 attention | flash-attn 变体（直接按 block 算，零 gather 拷贝） |
| prefill | 逐条 batch=1 | 合批 prefill（带 chunked prefill） |
| 抢占 | recompute only | recompute + swap |
| 采样 | 整批共用第一个请求的参数 | per-request |
| CUDA graph | 无 | 有（消 CPU 发射开销） |

## 6. v3 的候选方向

1. block table 常驻 GPU（每步的 `tables_batch` CPU 构造 + 拷贝是 profile 里的 HtoD 来源）；
2. 真 paged attention kernel（不 gather，按 block 表直接算——用 FlashInfer）；
3. 合批 prefill + chunked prefill；
4. per-request 采样参数；
5. CUDA graph 包 decode 步（0.6B decode 是 CPU 发射瓶颈，这是最大单项收益）。
