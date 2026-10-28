# v6 实验设计：并行配置权衡的验证方法

> 配合 PARALLEL_ANALYSIS.md（理论）——本文档回答"怎么实验"：每个理论论断对应的
> 实验设计、测量方法、预期曲线。bench 方法借鉴 vLLM 的 `vllm bench` 套件。

## 0. vLLM bench 方法借鉴（我们参考什么）

vLLM 的官方 benchmark 套件（`vllm bench throughput/latency/serve`）的关键设计：

| vLLM 的做法 | 我们采纳的方式 |
|---|---|
| `--input-len/--output-len` 可控合成负载（随机 token id） | `harness.py` 的 QUESTIONS + MAX_TOKEN_TIERS 改造为合成模式 |
| **随机路由敏感性**：MoE 负载用随机 token 会让 router 分布均匀 | 增加真实文本负载做对照（chat prompt 的 router 分布是训练后的真实分布） |
| `--batch-size` 扫描 | 我们的 batch 扫描循环（`speed.py --num-requests`） |
| NCCL 通信计时：`torch.cuda.Event` 包住通信 op | 微基准同样用 CUDA event（CPU 计时在异步通信下不准） |
| 产出 JSON + 统一后处理出表 | `results/*.json` + 汇总脚本 |

我们比 vLLM bench 多的部分：**通信/计算占比分解**（vLLM 不直接暴露——我们用
NCCL hook 或 event 对分别量出每步的通信时间和计算时间）。

## 1. 实验一：通信量随 batch 的变化曲线（验证 §2 的公式）

**问题**：EP 的通信量 ∝ token 数、TP 的 ∝ b×s×h——实验验证两条曲线的形状。

**方法**：
```
固定: 模型 Qwen3-Omni thinker, seq_len=128 (decode 阶段 s=1)
自变量: batch ∈ {1, 2, 4, 8, 16, 32, 64}
配置: TP2 / EP2 各跑一遍
测量: 每步通信时间（CUDA event 对包住 allreduce / all_to_all）
产出: 三条曲线（TP-comm, EP-comm, 计算时间）同图
预期: TP 通信随 b 线性增长且斜率大；EP 增长慢一个量级；两者在 batch≈X 交叉
```

**实现**（微基准，不需要完整引擎）：
```python
# benchmark/comm_bench.py
torch.cuda.Event 包住:
  TP: dist.all_reduce(hidden[b, 128, 2048])
  EP: dist.all_to_all_single(tokens[b, 2048] 分片)
# 每个配置跑 100 次取中位数, 报 min/p50/p99
```

## 2. 实验二：通信/计算占比分解（验证瓶颈归属）

**问题**：某配置下 decode 慢，是通信慢还是计算慢？

**方法**：在 v6 引擎的每步 forward 里插探针（可选开关 `--profile-comm`）：
```
每层: t_comm += all_to_all 的时间 (CUDA event 对)
     t_comp += 两次 gemm 的时间
产出: 每步的 (t_comm, t_comp, t_overlap, idle) 分解 + 占比饼图
判据: t_comm / (t_comm + t_comp - t_overlap) > 50% → 通信瓶颈
```

## 3. 实验三：并行配置扫描（TP/EP/PP/DP 的吞吐-时延权衡）

**问题**：12 卡怎么分最优？

**方法**：同负载跑配置矩阵，每个配置记录 (吞吐, TTFT p50/p99, TPOT, 显存/卡)：
```
配置矩阵（Qwen3-Omni 30B，单卡放得下所以 TP/EP 是研究性的）:
  DP2 (两节点各1卡)  / DP4 / DP12
  TP2 单机 / TP4 单机 / TP2×2 跨节点
  EP2 单机 / EP2 跨节点 / EP4
  PP2（micro-batch ∈ {2, 4, 8}——气泡敏感性）
负载: 3 组（短输出 32 / 中 128 / 长 512 token）× batch ∈ {8, 32}
判据: 吞吐/卡 的排序 + TTFT p99 排序（在线场景看尾时延）
```

**预期结论**（待实验验证）：
- DP 系的吞吐/卡 最高（零通信）但每卡显存占用最大
- EP 跨节点的通信成本 < TP 跨节点（实验一的曲线外推）
- PP 在 micro-batch 少时被气泡拖垮（吞吐/卡 最低），micro-batch 多时逼近 DP

## 4. 实验四：专家负载分布（P1 的调度研究基础）

**问题**：真实负载的 router 分布多不均？这决定 EP 的 all-to-all 是否有热点。

**方法**：跑真实 chat 负载，收集每层每步的 topk_idx → 专家命中直方图：
```
测量: 负载失衡系数 = max(expert_load) / mean(expert_load)，按 batch 大小扫描
预期: 小 batch 时失衡严重（几个热门专家被打爆），大 batch 趋近均匀
用途: P1 的调度依据——失衡 > 阈值时触发"溢出专家本地算"的降级策略
```

## 5. 实验五：与 vLLM 的同负载对比（外部基准）

**问题**：我们的 MoE 引擎离生产级多远？

**方法**：vLLM 用同模型同负载（`vllm bench throughput --model Qwen3-Omni`），
两边跑同一配置矩阵，差异分解为：kernel 质量（profile 算子）+ 通信实现（NCCL 版本/
拓扑感知）+ 调度（我们的 scheduler vs vLLM v1）。

## 6. 产出物

| 实验 | 脚本 | 输出 |
|---|---|---|
| 通信量曲线 | `comm_bench.py` | batch × 策略 的通信时间矩阵 + 交叉点 |
| 占比分解 | 引擎内 `--profile-comm` 探针 | 每步 (comm/comp/overlap/idle) 时间 + 饼图 |
| 配置扫描 | `parallel_sweep.py` | 配置 × 负载 的吞吐/时延/显存表 |
| 专家负载 | 引擎内 router 统计 | 失衡系数 vs batch 曲线 + 直方图 |
| vLLM 对比 | `serve_bench.py`（已有）+ vllm bench | 双方数字 + 差异归因 |

## 7. 实验纪律

- 每组数字跑 3 次取中位数（NCCL 有 JIT/拓扑探测的首跑效应，先 warmup 10 步）
- 显存/卡 同时记录（并行度提升常以显存利用率为代价，只看吞吐会误导）
- 跨节点实验先跑 `nvidia-smi` 确认两节点 GPU 空闲（共享集群的教训）
