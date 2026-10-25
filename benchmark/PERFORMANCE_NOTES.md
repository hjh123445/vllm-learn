# 性能思考笔记

记录性能相关的分析、实测与设计决策。方法学（怎么测）见 `benchmark/README.md`，本文件是"为什么这么设计、下一步改什么"。

---

## 1. 显存管理：现状、账本与 v2 方向

### 1.1 v1 现状（全部量出来的，不是算出来的）

| 量 | 数值 | 怎么量的 |
|---|---|---|
| 引擎总占用 | 4.95 GB | `torch.cuda.memory_allocated()`（分配器账本） |
| 其中 KV cache 池 | 3.76 GB | `tensor.numel() × element_size()`（张量自报） |
| 池子布局 | 每层各 2 片 `[8 slot, 8 kv头, 4096, 128]` bf16，共 56 片 | `attention.py` 的 `register_buffer` |
| 每 slot 成本 | 448 MB（115 KB/token） | 池 ÷ slot 数 |
| 实测利用率 | **7%**（12 请求混合负载） | 实际写入 token 数 × 每 token 字节 ÷ 池容量 |

三个结构性问题：

1. **slot 是不可分割的分配单位**：一条 50 token 的请求占一整个 448MB slot，
   12 条短请求也只能并发 8 条——池子的物理大小没变，逻辑容量差一个数量级；
2. **账本粒度是 slot 不是显存**：scheduler 准入只数 `free_slots` 个数，对显存是盲的；
3. **没有 OOM 防护**：cache 满额预分配，超了在启动时就炸；运行时没有水位检查、
   没有抢占。

### 1.2 为什么"预分配大池"是对的，问题只在"怎么切"

`cudaMalloc` 慢且碎片化，所以 vLLM / SGLang / TensorRT-LLM 全都是启动时一次性分配
整个 KV 池——**预分配是共同起点，工程差异全在池子的逻辑布局**：

| | v1（固定分区） | vLLM（分页） |
|---|---|---|
| 池子 | 同样预分配 | 同样预分配 |
| 切分单位 | 整条 slot（448MB） | 16 token 的 block |
| 50 tok 请求占 | 1 slot = 448MB | 4 block ≈ 7MB |
| 准入判据 | 剩余 slot 数 | 剩余 block 数（真实显存账本） |
| 驱逐收回 | 整个 slot（全有或全无） | 按 block（可只牺牲一条长请求） |
| 碎片 | 内部碎片极大（实测 93% 浪费） | 每序列平均浪费半个 block |

### 1.3 逐层分配 vs 中央一片：不是性能，是账本

v1 每层 Attention 各自持一片 cache（56 个张量、28 个物主、零个账本）。合成一片
`[28层, ...]` 后每层只是拿 `pool[layer_idx]` 视图，kernel 零区别，真正的收益：

- **block 分配跨层原子**：block #37 归谁 = `pool[:, 37]` 一个下标的事；逐层分配要靠
  代码保持 56 个张量同步，漏一层就是数据损坏；
- **显存账本唯一**：watermark / 容量估算 / OOM 预防需要"一个知道总量的对象"；
- 一次分配 vs 56 次。

vLLM 实际是折中：每层各一个 tensor（kernel 层面独立），但分配记账集中在
KVCacheManager——**关键是中央所有者，不是物理上一片还是 28 片**。

### 1.4 v2 设计草案：中央 KVCacheManager

```python
class KVCacheManager:
    """唯一的显存所有者。scheduler 只跟它要 block，不看物理布局。"""
    def __init__(self, num_layers, num_kv_heads, head_dim,
                 max_blocks, block_size=16, dtype, device): ...
        # self.k/v: [num_layers, max_blocks, num_kv_heads, block_size, head_dim]
        # self.free_blocks: list[int]
        # self.block_tables: {request_id: [block_idx, ...]}
    def can_allocate(self, n_tokens) -> bool      # 准入判据（替掉 free_slots 计数）
    def allocate(self, request_id, n_tokens)      # 准入时，全层原子
    def append_slot(self, request_id)            # 每步检查要不要新 block
    def free(self, request_id)                   # 驱逐/抢占，全层原子回收
    def watermark_ok(self) -> bool               # OOM 预防水位
```

池子大小从"拍脑袋的 max_requests × max_seq_len"改为算出来的：
`（总显存 − 权重 − 激活峰值 × watermark）÷ 每 block 字节数`。

---

## 2. 微基准方法论：四层金字塔

"纯理论推断 + 最后看 TTFT/TPOT"两端都是盲的：理论算不出 kernel 行为（局部性、发射
开销、索引拷贝路径），TTFT/TPOT 是总账（差 10% 无法归因）。实践是每层风险点插一个
可测的东西：

| 层 | 回答什么 | 测不了什么 |
|---|---|---|
| 纸面算术模型 | 该担心什么（decode 算术强度低 → 访存受限 → 该测 cache 读写） | kernel 真实行为 |
| 微基准 | 单个操作多贵、两种写法谁快（写之前就能测） | 交互效应（单 kernel 快 ≠ 端到端快） |
| 端到端 TTFT/TPOT | 最终裁判、整体多快 | 归因（差在哪不知道） |
| profile | 归因与验证（时间落在哪个算子、GPU 有没有空转） | 未实现的写法 |

微基准 vs profile 的分工：**微基准在写代码之前给预测，profile 在写完之后验证预测**。
显存这件事连微基准都不需要——`memory_allocated()` 直接读数。

---

## 3. 实测记录

全部可用 20 行合成张量脚本复现（见 §3.4），形状与 v1 一致（Qwen3-0.6B，decode 步
8 请求各 ~200 token）。

### 3.1 cache 读：行 gather vs block gather

| 写法 | µs/层/步 | 28 层/步 |
|---|---|---|
| v1 行 gather `pool[rows, :, :len]` | 36.2 | 1.01 ms |
| v2 block gather `pool[table]` | **26.8** | 0.75 ms |

**结论：paged gather 不但不亏，还快 25%——v2 最大的不确定性被提前消掉了。**

### 3.2 cache 写：逐行 python 循环 vs 向量化 scatter

| 写法 | µs/层/步 | 28 层/步 |
|---|---|---|
| v1 逐行循环 `for i: pool[int(rows[i]), :, pos, :] = k_new[i]` | 472 | **13.2 ms** |
| 向量化 scatter（线性下标一次写完） | **110** | 3.1 ms |

v1 实测 TPOT 49ms，光 write 循环就占 ~27%。两个开销来源：

1. **kernel 发射**：8 行 × 28 层 × 2 = 每步 448 次小 kernel 发射，每次搬 2KB，
   GPU 大部分时间在等 CPU 发下一条；
2. **设备同步**：`int(cuda_tensor[i])` 每行强制一次 GPU→CPU 同步——
   同一循环去掉同步（常量下标）只要 108µs，`int()` 同步占了 4/5 的时间。
   这也解释了 profile 里 `aten::item / _local_scalar_dense` 5940 次调用的来源。

**教训：任何在循环里对 CUDA 标量做 `int()/float()/.item()` 的写法，每次迭代都在
拖住整个 GPU。** 向量化写法（已验证 `torch.equal` 逐位一致）：

```python
# 每个目标元素算一个线性下标，一次 scatter 写完所有行
idx = 行偏移*H*P*D + 位置偏移*D + 头偏移*P*D + dim偏移
dst.view(-1)[idx] = src.reshape(-1)
```

### 3.3 profile 回顾（v1，4 请求 × 24 tok）

- **Self CPU 6.0s vs Self CUDA 0.76s**：87% 时间 GPU 在等 CPU——0.6B decode 是
  CPU 发射瓶颈，这是解读 v0/v1 对比数字的钥匙（v0 的"浪费行"几乎免费）；
- `aten::copy_` 占 CUDA 时间 74%：cache 写入 + 每步 `.to(device)` 的 HtoD 拷贝；
- `Memcpy HtoD (Pageable)` 510ms：输入/mask 每步在 CPU 构造再拷——应预分配复用。

### 3.4 待办优化清单（按预期收益排序）

| # | 改动 | 预期收益 | 风险 |
|---|---|---|---|
| 1 | v1 `KVCache.write` 向量化（§3.2 方案） | ~10ms/步 → 3ms，TPOT −8% | 低（已验证逐位一致） |
| 2 | 消 `.item()/.tolist()` 同步（采样结果留在 GPU） | 每次 decode 步省一次同步 | 低 |
| 3 | mask/输入张量预分配复用，消 HtoD | ~5% 总时间 | 低 |
| 4 | v2 paged KV + KVCacheManager（§1.4） | 逻辑容量 ×10+，OOM 防护 | 中（gather 已微基准验证） |
| 5 | 合批 prefill | 多请求同时到达时 TTFT | 低 |

---

## 4. benchmark 如何调度 v0 / v1（机制说明）

v0 和 v1 的模块名完全相同（`engine`、`layers`、`models`、`config`、`samplingParams`），
**同一个进程里不可能同时导入两套**——后导入的会命中 `sys.modules` 缓存拿到前一套。

所以 benchmark 的调度机制是**子进程 worker + sys.path 注入**：

```
speed.py / bubble.py / accuracy.py（驱动进程，不 import 任何引擎）
    │ subprocess.run([sys.executable, harness.py, "--engine", "v0"])
    ▼
harness.py（worker 进程）
    sys.path.insert(0, 仓库根/v0)          # 把 v0 目录插到 sys.path 最前
    from engine.scheduler import Scheduler  # → 实际解析到 v0/engine/scheduler.py
    from engine.model_runner import ModelRunner
```

- `--engine v0` 和 `--engine v1` 的唯一区别就是 insert 的目录；
- 每个 worker 是干净进程，`sys.modules` 无缓存，导到谁就是谁；
- 驱动进程通过 worker 的 stdout（JSON）收集结果，跨进程比较（accuracy T3 的
  v0 vs v1 对比就是这样做的：两个 worker 各自跑，驱动进程对 JSON）。

代价：每次 worker 都要重新加载一遍权重（~15s），accuracy 全套跑一次约 5 分钟。
如果以后测试项多了，可以换成"每个测试复用常驻 worker 进程"的模式，但 v0/v1
互斥导入这个约束永远存在，进程隔离是唯一干净的解法。
