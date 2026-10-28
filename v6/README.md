# v6：多卡并行（TP / EP / PP 及组合）

建立在 v5（在线推理引擎）之上。从单卡到双节点 12 张 A800，系统实现并实测对比了
TP/EP/PP 三种并行及组合，产出了通信量公式、单机矩阵、跨节点崩塌曲线和组合选型决策树。

## 小版本演进（每个小版本有独立文档，见 MILESTONES.md）

| 小版本 | 主题 | 状态 |
|---|---|---|
| v6.0 | Qwen3-Omni-30B MoE 单卡（router + grouped experts + fp32 对拍 0.0） | ✓ |
| v6.1 | TP2/TP4/TP8 单机 + 跨节点 TP8（Megatron 切分） | ✓ |
| v6.2 | EP2 单机 + 跨节点（专家对半 + all_reduce 聚合） | ✓ |
| v6.3 | DeepSeek-V4 模型文件（MLA + sqrtsoftplus + FP8 反量化） | 待做 |
| v6.4 | 调度集中化（rank0 决策 + broadcast，解 EP>2/PP lockstep） | 待做 |

## 核心实测结论（数据全在 PARALLEL_ANALYSIS.md）

**单机 8 卡矩阵**（Qwen3-Omni-30B，3 请求 × 48 tok greedy）：

| 配置 | 吞吐 | 输出 |
|---|---|---|
| EP2 | **11.3 tok/s** | ✓ |
| TP2 | 10.7 | ✓ |
| TP4 | 10.4 | ✓ |
| TP8 | 8.3 | ✓（小 batch 下 TP 越多越慢——通信稀释计算） |

**跨节点对比**（bond0 TCP，节点 56 IB 端口全 DOWN 是环境约束）：

| 配置 | 吞吐 | 相对 |
|---|---|---|
| EP2 单机 | 11.8 | 1× |
| **EP2 跨节点** | **9.3** | **0.79×（只损 21%）** |
| TP4 单机 | 10.0 | — |
| TP8 跨节点 | 1.2-3.7 | **崩塌 69-88%** |

**组合最优**（§10.3）：最优组合 = f(显存矛盾位置, batch 大小, 链路带宽)。
12 卡推荐 EP4×3 副本 DP。

## 文档索引

- [PARALLEL_ANALYSIS.md](PARALLEL_ANALYSIS.md) —— 通信量公式 / 重叠分析 / 决策树（§2-4）/
  跨节点崩塌实测（§8-9）/ 单机矩阵与组合探讨（§10）
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) —— 五个实验的设计方法
- [MILESTONES.md](MILESTONES.md) —— **每个小版本的独立工作文档**（现象→排查→根因→解决）

## 代码结构

```
engine/parallel.py     TP/EP 切分 + 通信原语（kv 复制 / clone 教训 / 组管理）
tp_worker.py           TP worker（独立进程——spawn+Manager 的坑见 MILESTONES）
ep_worker.py           EP worker
pp_worker.py           PP worker（层切分 + p2p）
parallel_worker.py     TP×EP 组合 worker
run_tp.py / run_ep.py / run_pp.py / run_parallel.py / run_tp_crossnode.py   启动器
comm_bench.py          TP vs EP 通信微基准
```
