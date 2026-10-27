# v6 设计：双节点 A800 上的 MoE 并行推理

> 状态：设计定稿，实施分阶段。双节点硬件已实测验证（§1）。
> 项目定位（§0）是这份文档的核心——其余版本的技术优化都是基本功，这里回答"为什么做这个项目"。

---

## 0. 项目定位：解决什么具体问题

**场景**：两张 A800-80G（分居两个节点，IB 互联），推理一个前沿 MoE 模型
（Qwen3-30B-A3B 级别，激活 3B/总参 30B）。单卡放得下但 decode 带宽受限于单卡 HBM；
两张卡怎么用，取决于三个**具体的、可度量的问题**——这是项目的三个研究点：

### P1 调度问题：MoE 的负载不均如何影响 batch 内通信

Dense 模型的 batch 内所有序列共享同一份权重；MoE 每个 token 各自路由到 8/128 个专家，
**batch 越大专家分布越不均**，EP（专家并行）下 all-to-all 的通信量与热点专家的负载失衡
直接相关。要回答的具体问题：
- 给定 batch 组成，**dispatch 前预测** expert 负载（token→expert 的路由可以离线统计），
  scheduler 据此决定"这个 batch 值得跨节点 EP 还是本地多算"；
- 验证手段：路由统计直方图 + 负载失衡系数（max/mean expert 负载）随 batch 大小的曲线，
  与实测 all-to-all 耗时做相关分析。

### P2 通信问题：40Gb IB 上 TP/EP 的真实成本与选型决策树

双节点 TP 的每层 allreduce 要跨 IB；EP 的每层 all-to-all 同样跨 IB。
**在 40Gb/s（≈4.5GB/s 单向）的实测约束下**，通信成本模型：
- TP2 allreduce/层 ≈ 2×(batch×seq×hidden×2B) / 4.5GB/s（ring：2 倍数据量传输）
- EP all-to-all/层 ≈ token 数 × hidden × 2B × 2（dispatch+combine）
要回答的具体问题：**多大的 batch 下 TP/EP 的通信开销摊得薄**——在 IB 带宽远低于单卡 HBM
（2TB/s）的机器上，这个决策与 8×GPU 同机完全不同（同机 NVLink 400GB/s 时答案是无脑并行）。
验证手段：NCCL 通信微基准（per-layer allreduce/all-to-all 实测）× batch 大小扫描曲线。

### P3 自适应调度：profile 驱动的动态预算

静态配置（max_num_batched_tokens / max_num_seqs / 池子大小）在不同模型/负载下次优：
8B dense 在 util 0.3 抢占风暴、MoE 激活参数随路由波动。要回答的具体问题：
- 池子 block 数由 warmup 实测（v2 已做），**max_num_batched_tokens 也应实测**：
  warmup 时扫描 chunk 大小，找到 decode TPOT 恶化 ≤5% 的最大 prefill chunk——
  这就是 chunked prefill 的时延/吞吐最优点，不需要人为配置；
- expert 负载统计进 scheduler：MoE 下准入预算用"激活参数量×batch"而非"token 数"。

**项目一句话**：在低带宽（40Gb IB）双节点上，研究"通信感知的 MoE 推理调度"——
调度决策（batch 组成/并行策略）由实测的通信成本和专家负载统计驱动，而非静态配置。
三个研究点各自有微基准/消融实验可验证。

---

## 1. 双节点硬件实测（2026-10-27 实测）

| 项 | 实测值 |
|---|---|
| 本机 → 172.28.30.56 ping | 通，**RTT 0.2ms**（同机房） |
| SSH (22) | 通，但**免密未配置**（需要把本机公钥加到对面） |
| 本机 IB HCA | 有，**40 Gb/s（4X QDR）** |
| NCCL | 2.27.7（torch 2.9.1 自带，IB transport 可用） |
| 对面节点 GPU | 待 SSH 通了确认（用户说也是 A800） |

**结论**：多节点 NCCL 的硬件链路成立（IB 40Gb ≈ 4.5GB/s 单向，比 TCP 快 ~4 倍，
RTT 0.2ms 对 allreduce 延迟可忽略）。**阻塞项：SSH 免密**——这是唯一需要用户配置的前置条件。

## 2. 并行策略选型（决策有数据依据，不是"都支持"）

| 策略 | 通信模式 | 何时用（本项目硬件下的结论） |
|---|---|---|
| **DP（数据并行）** | 零通信（各跑各的请求） | 默认首选。两个节点各起一个引擎，负载均衡分发请求，**吞吐直接 ×2，无任何跨机通信**。验证成本最低 |
| **EP（专家并行）** | 每层 all-to-all（dispatch+combine） | MoE 原生并行：专家切两半放两节点。**通信量 ∝ token 数**，与 hidden 无关，大 batch 下摊薄。研究点 P1/P2 的主战场 |
| **TP（张量并行）** | 每层 allreduce ×2 | 单卡放不下模型时才用（本项目 30B-A3B bf16 61GB 单卡放得下，**不需要 TP**）；且 TP 通信量 ∝ batch×seq×hidden，40Gb IB 下代价最大 |

**结论**：v6 = DP 为骨架 + EP 为研究核心，TP 仅作对比基线（证明"40Gb IB 上 TP 不划算"
本身就是 P2 的一个可验证结论）。

## 3. 架构

```
node-0 (172.29.226.143)                 node-1 (172.28.30.56)
┌────────────────────────┐              ┌────────────────────────┐
│ LLMEngine (v5 引擎)     │              │ LLMEngine (v5 引擎)     │
│ scheduler + 本地专家    │◄── IB 40G ──►│ scheduler + 远半专家    │
│ MoE 层: EP all-to-all   │  (dispatch/  │                        │
│                        │   combine)   │                        │
└────────────────────────┘              └────────────────────────┘
        ▲                DP 层：请求路由（负载均衡，按节点 KV 占用）▲
        └──────────── HTTP server (v5) 分发到两节点 ────────────────┘
```

- **DP 层**：server 按"两节点各自 KV 占用"加权分发（不是简单轮询——两节点速度可能不同），
  最简单的两节点扩展，单独可验证（吞吐 ×2 直测）。
- **EP 层**：MoE 每层 gate 在 token 分发后做 all-to-all，把 token 送到其专家所在节点，
  算完 combine 回来。**路由统计**（P1）在 gate 输出上采样统计，喂给 scheduler。
- 通信原语：torch.distributed + NCCL（`all_to_all_single` 带 API），不需要自写。

## 4. 自适应调度（P3 的实现）

```
warmup 阶段（引擎构造时，一次性）：
  1. 权重实测 → 池子 block 数（v2 已做）
  2. chunk 大小扫描 {512, 1024, 2048, 4096, 8192} × 各跑 8 步 decode
     → 取 decode TPOT 恶化 ≤5% 的最大 chunk 为 max_num_batched_tokens
     （chunked prefill 的时延/吞吐最优点是量出来的，不是默认 16384）
  3. MoE: 路由分布采样 → 专家负载直方图 → 动态 max_num_seqs
     （热点专家到达阈值的 batch 组成 → 准入上限）
运行时：
  - expert 负载滑动窗口统计 → EP dispatch 前预测本批 all-to-all 规模
  - 超阈值 → 该批降级为"本地专家多算"（dispatch 时把溢出 token 路由到本地备用专家）
```

## 5. MoE 模型选型

| 模型 | 总参/激活 | bf16 大小 | 专家数 | 备注 |
|---|---|---|---|---|
| **Qwen3-30B-A3B**（推荐） | 30B/3.3B | ~61GB | 128 选 8 | 单卡放得下（EP 研究不需要 TP）；**自带 MTP 层**（v7 投机解码直接用，见 §7）；HF 格式齐全 |
| GLM-4.5-Air | 106B/12B | ~212GB | 160(共享1+路由) | 单卡放不下 → 必须跨节点 EP+TP，通信研究更"重"；tokenizer/格式复杂 |
| DeepSeek-V2-Lite | 16B/2.4B | ~32GB | 64 选 6 | 轻量热身用（先在它上面把 EP 跑通再上 30B） |

建议：**Qwen3-30B-A3B**（下载 61GB，磁盘 461GB 够；先 DeepSeek-V2-Lite 热身可选）。

## 6. 分阶段路线图（每阶段有验证门槛）

| 阶段 | 内容 | 验证门槛 |
|---|---|---|
| 6.0 | MoE 模型支持（单卡）：gate/router/expert 层 + top-k 路由 | fp32 HF 对齐 100%（同 T4 标准） |
| 6.1 | 自适应预算（单卡）：chunk 扫描/池子/动态阈值全 profile 驱动 | 同负载下优于静态配置的消融 |
| 6.2 | DP 双节点：请求按 KV 占用加权分发 | 2× 吞吐实测 ≥1.8× |
| 6.3 | EP 双节点：all-to-all + 专家切分 | P2 通信微基准 + batch 扫描曲线 |
| 6.4 | P1 完整闭环：路由统计驱动调度 | 负载失衡场景的吞吐提升消融 |

**需要用户决策/配置的**：(a) 172.28.30.56 的 SSH 免密（唯一硬阻塞）；(b) 确认 MoE 模型选型
（建议 Qwen3-30B-A3B，需下载 61GB）；(c) 对面节点的 GPU 型号和 python 环境确认。

## 7. v7 预告：投机解码（当前 SOTA 调研）

| 算法 | 需要什么 | 加速比（文献） | 我们的适配性 |
|---|---|---|---|
| **EAGLE-3**（当前 SOTA，2025） | 训练一个单层 draft head（复用主模型隐状态 + 多尺度特征） | 3-5× | 需要训练 draft head（一次小训练）；Qwen3 有 MTP 层可免训练 |
| **MTP**（DeepSeek-V3/Qwen3 自带） | 模型自带的 Multi-Token Prediction 层，推理时当 draft head | 2-3× | **首选**：Qwen3-30B-A3B 自带 MTP 权重，零训练成本 |
| Medusa（2024） | 每层挂多个解码头 + tree attention | 2-3× | 实现最重（tree mask）；EAGLE 系列已胜出 |
| 自回归 n-gram（最简） | 无需模型改动，纯统计 | 1.2-1.5× | 热身用，验证 verify 流程 |

**v7 路线**：Qwen3-MTP 层做 draft（零训练）→ 目标验证式投机（draft k 个 token →
并行 verify → 接受长度分布统计）。关键实现点：(a) verify 的并行化就是 v1 的平铺批
（把 draft tokens 拼进 batch 一次前向）；(b) **接受率统计**与动态 draft 长度（接受长度
的滑动窗口 → 自适应 k）——这是投机解码里最值得自己做的"调度"研究，与 v6 的 P3 一脉相承。

## 8. 与已有版本的关系

v0-v5 的全部优化（paged KV/context/triton kernel/在线服务）都是 v6 的地基——它们解决的是
"单卡上跑得快"，v6 在此之上叠加"多卡/多节点上跑得聪明"。基本功体现在：
triton kernel 写作（v4）→ MoE 的 expert kernel 沿用同一套；warmup 显存预算（v2）→ 自适应
调度（P3）的直接扩展；平铺批（v1）→ 投机 verify（v7）的复用。
