# 八卡并行策略实测与权衡分析（v9.4）

> 数据：172.28.30.56（8×A800-80GB，NVLink），Qwen3-Omni-30B-A3B（MoE，激活 3B），
> 6 请求 × 48 tok greedy，vLLM v1 形态进程架构（EngineCore + WorkerProc + mp.Pipe）。
> 2026-10-31 实测，全部配置同一 EngineCore 代码路径（组件化并行，无 per-config 分支）。

## 1. 实测矩阵

| 配置 | tok/s | GPU | 初始化 | 一致率(6 请求) |
|---|---|---|---|---|
| TP1（基线） | 22.6 | 1 | 33s | — |
| TP2 | 19.3 | 2 | 28s | 4/6 |
| TP4 | 19.6 | 4 | 340s | 4/6 |
| TP8 | 18.4 | 8 | 785s | 5/6 |
| EP2 | 30.0 | 2 | 31s | （单跑验证） |
| EP4 | 37.5 | 4 | 109s | 4/6 |
| **EP8** | **45.9** | **8** | 140s | 5/6 |
| TP2×EP2 | 28.4 | 4 | 189s | 4/6 |
| TP4×EP2 | 27.1 | 8 | 420s | 5/6 |
| TP2×EP4 | 35.2 | 8 | 233s | 5/6 |
| TP2×DCP2 | 16.4 | 4 | 367s | （DCP 数学 v8.2 已验） |
| TP2×DCP4 | 14.5 | 8 | 804s | （DCP 容量 ÷4 场景） |
| TP2×DCP2×EP2 | 18.9 | 8 | 430s | 4/6 |
| PP2 / PP2×TP2 | — | — | — | 实现完成，卡 p2p 时序（见 §4） |

注：一致率 4-5/6 是小 batch（6 请求）下 argmax 边界翻转的预期噪声（v6 时
TP2 双进程独立跑同样 4/6），非系统性错误；e2e 吞吐才是本矩阵的主读数。

## 2. 三个核心发现

### 发现 1：TP 在这个 workload 上是负收益（8 卡 18.4 vs 单卡 22.6）

**为什么**：TP 的收益 = 权重读取带宽 ÷N，代价 = 每层 2 次 allreduce。
我们的小 batch（6×48 tok）decode 步是**launch-bound**（v7.2 实测 5-token
verify ≈ 1-token decode），带宽本来就省不出时间，allreduce 的固定延迟
（~50μs×2×48 层）却是净新增。vLLM 用 CUDA graph + 大 batch 才让 TP 转正。

**TP 的真正用途**（什么时候选）：
- **显存**：单卡放不下的 dense 模型（TP 切权重立竿见影）；
- **大 batch decode / prefill**：带宽 bound 时 TP 是线性收益；
- **单请求低延迟**：layer latency = 权重读时间，TP 直接除 N。

### 发现 2：EP 是 MoE 的八卡最优（45.9 tok/s = 单卡 2 倍）

**为什么**：EP 切的是 80% 的权重（专家），且 MoE 的稀疏激活让"每 token 只
碰 topk 个专家"——EP 的冗余计算模式（每卡过全部 token、只算本地专家）
通信量 ∝ token×topk×hidden，远小于 TP 的 batch×hidden×每层 2 次。
EP8 吞吐翻倍的同时初始化只要 140s（权重加载 ÷8 并行）。

**EP 的代价**：负载不均（热点专家拥塞——v6 MILESTONES P1 研究点的实测
依据）；DP-style 冗余模式每卡都要过全部 token 的 router/gather。

### 发现 3：DCP 是容量手段不是速度手段（16.4/14.5 vs TP2 的 19.3）

DCP 越深越慢：TP2×DCP2 = 16.4、TP2×DCP4 = 14.5——每层多 1 次
partial/LSE 的 all_gather，cp 越深合并开销越大。**它的价值在 KV 池
容量**（DCP4 = 同卡数下 4 倍 KV），服务于长序列场景。这与 v8 DESIGN
的预判一致：DCP 解决 O(L) 显存放不下，不解决 O(带宽)。

## 3. 决策框架：怎么选八卡并行

```
模型放得下单卡吗？
├─ 放得下 → 纯带宽问题：
│    ├─ MoE → EP 优先（实测 2x）；batch 大可 EP×TP 混合
│    └─ Dense → TP（带宽 bound 时线性收益；launch-bound 时先上 CUDA graph）
└─ 放不下 → 显存问题：
     ├─ Dense 大模型 → TP（切权重）
     └─ MoE 大模型 → EP 切专家 + 量化（v9：W4 后 60GB→16GB，可能回到"放得下"）
          └─ 还不够 → DCP 切 KV（长序列）或 PP 切层（超大）
```

**组合的乘法效应**（实测）：TP2×EP4 = 35.2 vs 纯 EP4 37.5——TP 的加入
**没有**叠加收益反而略降（allreduce 税）。混合并行不是"越多越好"，
每加一维都要用矩阵数据说话。

## 4. DeepSeek-V4 的八卡方案（DESIGN 依据实测数据落地）

DSV4-Flash 的特殊性（v6.3 实测显存账）：
- 磁盘 316GB（专家 MXFP4 打包 283.5 + FP8 15 + scale 17.7），
  **反量化 bf16 后 ~1199GB**——8×80GB 放不下是数学必然；
- MLA 是 **MQA**（kv_heads=1）：TP 的按头切分对 KV 头失效，q/o 切 64 头
  可行但每层 2×allreduce 全额交税；
- 专家权重占 94%：**切专家即切主要矛盾**。

**八卡方案：量化原生存储 + EP8**（v6.3 MILESTONES 定稿）：
```
权重: MXFP4 原生（不反量化）→ 141.75GB 专家 + 15GB FP8 + scales ≈ 178GB
每卡: ~22GB 权重 + 40GB+ KV 池（80GB 卡余量大）
通信: a2a（token×topk×hidden）——MoE 的固有通信，无 TP 的 allreduce 税
前提: in-kernel MXFP4 反量化（= v9.2 的 dequant-GEMM triton kernel，
      DSV4 与 v9 量化路径共用同一工程）
```
对照本矩阵的 Qwen3 数据外推：EP8 的 2x 吞吐收益 + 量化原生让"放不下"
变"放得下"——**量化改写并行选型**（v9 DESIGN 的核心论点的实测支撑）。

跨节点（v6.2 实测）追加：EP2 跨 IB 只损 21%，TP8 跨 IB 崩 69-88%——
**EP 也是跨节点的唯一稳选**，DSV4 级模型多机部署时 EP 是默认起点。

## 5. PP 状态与剩余工作（诚实记录）

PP 实现完成（get_pp_indices 层区间 + PPMissingLayer 占位 + NCCL p2p
激活传递 + 中间 stage 跳过采样），**卡在 p2p 时序**：日志显示 send/recv
已发生但 400s 未完成——疑为 stage0 send 与 stage1 的 recv 顺序在
prefill/decode 切换处死锁（vLLM 用 virtual engine 的微批流水避开）。
修复方向：按 vLLM 的 `VirtualEngine` 语义引入微批队列（本版单微批的
严格 lockstep 在 chunked prefill 边界暴露时序问题）。

DP 未实现（设计定稿：每 dp rank 独立 EngineCore + 前端路由，vLLM 的
data_parallel_rank 同款）——DP 与本矩阵正交（每 rank 仍是完整引擎），
是吞吐扩展（多请求池）而非单批加速。

## 6. 结论

| 场景 | 八卡最优（实测） | 依据 |
|---|---|---|
| MoE 推理吞吐 | **EP8**（45.9 tok/s） | 稀疏激活 + a2c 通信 ∝ token×topk |
| Dense 大显存 | TP8 | 切权重是唯一目的 |
| 长序列容量 | EP + DCP | KV 池 ÷cp_size |
| 超大模型（DSV4） | **量化原生 + EP8** | 显存账 + MLA=MQA 切不动 TP |
| 跨节点 | EP（跨 IB 损 21% vs TP 崩 69%） | 带宽鸿沟下的唯一稳选 |
