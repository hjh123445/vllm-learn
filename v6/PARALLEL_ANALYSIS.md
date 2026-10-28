# v6 并行策略系统分析：TP / EP / DP / PP 的定量对比

> 在 12×A800-80G（本机 4 + 节点 56 8 卡）、IB 40Gb/s 的实测约束下。
> 本文档是 DESIGN.md §2 的展开：通信量公式 + 计算通信重叠 + 选型决策树。

## 1. 四种并行的本质

| | 切什么 | 通信模式 | 通信量 | 什么时候用 |
|---|---|---|---|---|
| **DP** | 请求 | 零通信 | 0 | 两节点各跑完整副本，请求分发。**吞吐 ×2，无任何跨机通信** |
| **TP** | 层内权重矩阵 | 每层 2 次 allreduce | 2×b×s×h×2B / 层 | 单卡放不下模型或单卡算力不足 |
| **EP** | MoE 专家 | 每层 2 次 all-to-all | token 数 × h × 2B × 2 / 层 | MoE 原生并行；专家数多时均摊最好 |
| **PP** | 层（深度） | 1 次 p2p send / 层边界 | b×s×h×2B / 层边界 | 单卡放不下模型且 TP 通信太贵时的替代 |

## 2. 通信量公式（本项目硬件的实测参数）

符号：b=batch, s=seq_len, h=hidden_size, L=num_layers, N=GPU 数, **IB 带宽 = 40Gb/s ≈ 4.5 GB/s 单向**

### TP（张量并行）

每层 2 次 allreduce（attn 输出 + MLP 输出），ring allreduce 传输 2×(N-1)/N × 数据量：

```
每层通信量 = 2 × 2×(N-1)/N × b×s×h×2B
           ≈ 4 × b×s×h×2B   （N=2 时 2×(1/2)=1, 即 2×1×bsh×2B = 2bshB/层）

TP2 跨节点（IB 40Gb）：通信时延/层 = 2×b×s×h×2B / 4.5GB/s
  b=1,s=1,h=5120: ≈ 4.5μs/层 × 40 层 = 180μs/步
  b=8,s=512: ≈ 37ms/步  ← decode 时已经不可忽略
```

### EP（专家并行）

每层 2 次 all-to-all（dispatch + combine），通信量 ∝ token 数 × hidden：

```
每层通信量 = 2 × tokens × h × 2B
           （token 送到专家所在卡算完送回来）

EP2 跨节点：通信时延/层 = 2 × tokens × h × 2B / 4.5GB/s
  decode b=8, s=1: ≈ 2×8×5120×2B / 4.5GB/s = 36μs/层 × 40 MoE 层 ≈ 1.4ms/步
  prefill b=8, s=512: ≈ 9.3ms/步  ← chunked prefill 的必要性
```

**关键洞察：EP 的 decode 通信（36μs/层）远小于 TP 的（b=8 时 37ms/层）**——因为 EP 通信量 ∝ token 数（每个 token 只发给它的 6 个专家），而 TP 通信量 ∝ batch×seq×hidden（每个 token 的全部隐状态都要同步）。

### PP（流水线并行）

层边界 1 次 p2p send，通信量 = b×s×h×2B / 层边界。**通信量与 EP/TP 同量级但次数更少**（L/N 次 vs L 次），代价是**流水线气泡**：

```
气泡率 = (N-1) / (N + N_micro - 1)    （N=stage 数, N_micro=micro-batch 数）

PP2（2 stage, 8 micro-batch）: 气泡率 = 1/9 ≈ 11%
PP2（2 stage, 1 micro-batch）: 气泡率 = 1/3 ≈ 33%
```

PP 的唯一通信是层边界的 p2p——**不需要 allreduce**（每层只在切分点通信一次），通信次数 = N-1 次/步 vs TP 的 2L 次/步。

### DP（数据并行）

零通信。两节点各跑独立引擎，请求由 server 层分发。

## 3. 计算通信重叠分析

### TP：不可重叠（串行依赖）

```
attn: x → qkv_proj → attention → o_proj → allreduce → MLP → allreduce → 下一层
                                              ↑ 必须等 o_proj 算完才能 allreduce
```
TP 的 allreduce 在关键路径上，**无法与计算重叠**（下一层的输入依赖 allreduce 的输出）。

### EP：dispatch 可与路由计算重叠

```
router: x → gate → top-k 选择 → dispatch (all-to-all) → expert FFN → combine (all-to-all) → 下一层
                                ↑ dispatch 可以在路由做完后立即发出
                                  （此时非目标卡的 expert 还在算上一批，dispatch 的传输与之重叠）
```
EP 的 dispatch 发出后本卡可以继续算**留在本卡的 token**——部分重叠。

### PP：天然重叠（流水线推送）

```
stage 0: [mb0 fwd] [mb1 fwd] [mb0 bwd] ...     ← 计算 mb1 时 mb0 的输出正在传给 stage 1
stage 1:         [mb0 fwd] [mb1 fwd] ...        ← stage 0 算完 mb0 后马上开始 mb1
```
PP 的 p2p send 与下一 stage 的计算**天然重叠**（只要 micro-batch 足够多）。

### DP：完全无需重叠

零通信。

## 4. 本项目硬件下的选型决策树

```
单卡放得下模型？
├─ 是 → 需要多卡吞吐？
│        ├─ 是 → DP（零通信，吞吐 ×N，无脑最优）
│        └─ 否 → 单卡
└─ 否 → 模型多大？
         ├─ 1 卡 < model < N 卡 → TP? EP? PP?
         │    ├─ MoE 模型 → EP 首选（通信量 ∝ token，比 TP 的 ∝ b×s×h 小几个量级）
         │    │    └─ EP 后单卡仍放不下 → 叠 TP2
         │    └─ Dense 模型 → TP vs PP 的选择：
         │         TP: 通信多但无气泡、延迟低 → 小 batch 在线推理
         │         PP: 通信少但有气泡       → 大 batch 离线吞吐 / 层数多
         └─ 模型 > N 卡 → TP + PP 混合
```

**本项目的结论**（12×A800，IB 40Gb）：

| 场景 | 最优策略 | 理由 |
|---|---|---|
| Qwen3-30B-A3B（61GB bf16） | DP×2（双节点各一份） | 单卡放得下（61GB < 80GB），零通信直接 ×2 吞吐 |
| DeepSeek V4.1-Flash（510GB） | EP×4~8 + DP×2~3 | 384 专家分散到 4~8 卡（每卡 48~96 专家，约 64~128GB/卡）；DP 再跨节点 ×2 |
| 验证 TP vs PP | TP×2 / PP×2 | DeepSeek-V4.1 单层很大（hidden 5120, MoE 中间 2304），TP 通信重；PP 层间通信轻但有气泡 |

## 5. 本机两个 MoE 模型的并行方案（已确认可用，无需下载）

### 5.1 Qwen3-Omni-30B-A3B-Instruct（66GB bf16，@/data/jingjiang3/res/AI-ModelScope/）

thinker LLM 主体（本项目只做 thinker 的文本推理）：
```
hidden_size = 2048, num_layers = 48（decoder_sparse_step=1 全 MoE）, head_dim = 128
num_experts = 128, num_experts_per_tok = 8, shared_expert_intermediate_size = 0（无共享专家）
moe_intermediate_size = 768, GQA: 32 q头 / 4 kv头
bf16 权重 66GB → 单卡放得下（80GB），但要留 KV cache 池 → 双卡刚好
```

**方案 A（开发验证）：TP2 单机**——每卡 ~33GB 权重 + 充足 KV 池。TP 的通信量
（b×s×2048×2B×2/层）在 NVLink 同机双卡上可忽略，适合先把 MoE 前向写对再研究跨节点。

**方案 B（v6.2+ 研究）：EP2/EP4 跨节点**——128 专家分 2 卡各 64（或 4 卡各 32）。
每层 dispatch+combine 通信 = 2 × tokens × 2048 × 2B，decode b=8 时仅 64KB/层，
43 层 ≈ 2.8MB/步 → 40Gb IB 上 **0.6ms/步**（可忽略）——EP 在小 hidden MoE 上通信极便宜。

### 5.2 DeepSeek-V4-Flash（149GB FP8，@/data/xmhuang6/deepseek/，只读）

新架构全家桶——v6 的终极目标：
```
model_type = deepseek_v4, 43 层, hidden = 4096
MLA: q_lora_rank = 1024, qk_rope_head_dim = 64, head_dim = 512, kv_lora_rank 压缩注意力
MoE: 256 routed experts top-6 + 1 shared, moe_intermediate = 2048
路由打分: sqrtsoftplus（新函数，非 softmax）
FP8 block 量化 (128×128) → 149GB
```

**方案（v6.3）：EP4 单机 4 卡**——每卡 ~37GB 权重（149/4）+ KV 池。
每层 all-to-all = 2 × tokens × 4096 × 2B，decode b=8 时 128KB/层，43 层 ≈ 5.5MB/步。
**MLA 不切分**（q_lora_rank 压缩后 attention 极小，每卡独立算）。
FP8 权重反量化为 bf16 计算（loader 里做 dequant，或保持 FP8 存储 + kernel 反量化——
v6.3 先做前者，kernel 级 FP8 是 v6.4）。

**v7 投机解码**：DeepSeek V4 自带 MTP 权重（config 的 MTP 层），draft head 零训练。
接受率 70% + k=3 → 净加速 ≈ 2.4×（EP 下计入通信增加 ≈ 1.8×）。

### 5.3 实现顺序（由易到难，每步有验证门槛）

| 步 | 内容 | 模型 | 验证 |
|---|---|---|---|
| 6.0 | MoE 层（router + grouped expert + top-k）单卡 | Qwen3-Omni thinker | fp32 HF 对齐 |
| 6.1 | TP2（单机双卡，attention + expert 列切） | Qwen3-Omni | 单卡逐 token 等价 |
| 6.2 | EP2 跨节点（专家对半 + all-to-all） | Qwen3-Omni | TP2 逐 token 等价 + 通信微基准 |
| 6.3 | DeepSeek-V4 模型文件（MLA + sqrtsoftplus + FP8 反量化） | DeepSeek-V4-Flash | fp32 HF 对齐（dequant 后） |
| 6.4 | EP4 + TP2 混合并行 + 通信重叠 | DeepSeek-V4-Flash | 12 卡吞吐 + 通信/计算占比 profile |
| 6.5 | PP 对比基线（证明 EP 优于 PP/TP 的定量数据） | DeepSeek-V4-Flash | 三策略同负载对比表 |

## 6. 通信量汇总表

| 策略 | 每层通信量 | 通信次数/步 | 可重叠？ | 本项目 IB 实测成本/步 |
|---|---|---|---|---|
| DP×N | 0 | 0 | — | 0 |
| TP×2 | 2×b×s×h×2B | 2L = 80 | ❌ 串行依赖 | b=8,s=1: **0.36ms** |
| EP×8 | 2×tokens×h×2B | 2L = 80 | 部分（dispatch 与本地计算） | b=8,s=1: **0.07ms** |
| PP×2 | b×s×h×2B | L/N = 20 | ✓ 天然重叠 | b=8,s=1: **0.18ms**（+气泡 11-33%） |

## 7. 为什么 EP 是 MoE 的最优选择（定量论证）

TP 每层同步**全部 token 的全部隐状态**（通信量 ∝ b×s×h）。
EP 每层只把每个 token 发给它的 6 个专家所在卡（通信量 ∝ b×s×h，但经过 all-to-all 的
permutation 后每卡只收到自己专家对应的 token 子集——**均摊后实际通信量是 TP 的 1/N**）。

具体地，EP×N 的 dispatch：每卡发出自己 batch 的 token 到对应专家卡，
实际传输 ≈ b×s×h×2B / N × N = b×s×h×2B（总量相同但 per-hop 只有 1/N 在跨卡链路上）。

但 EP 的**真正优势**是计算端的：384 个专家分散到 N 卡后每卡只算自己的 48 个专家的 token，
而 TP 的每卡要算全部 token 的 1/N 切片（每卡都要碰每个 token）。

对于 MoE 模型（激活参数 << 总参），**EP 让每卡只碰激活参数的 1/N，TP 让每卡碰全部 token 的 1/N**
——EP 的本质优势是计算量跟激活参数走，TP 的计算量跟总 token 走。
