# v9.4：vLLM v1 形态的进程架构 + 全并行矩阵

> 对标源码：/data/ryxu/vllm（v1 目录）。本文档是读源码后的设计映射，
> 不是抄源码——接口对齐 vLLM 语义，实现用我们的组件栈。

---

## 0. vLLM v1 的真实架构（源码确认的事实）

```
进程 1: 前端（AsyncLLM / HTTP SSE）           ← v1/engine/async_llm.py
   ↕ ZMQ socket（EngineCoreClient: MPClient）  ← v1/engine/core_client.py
进程 2: EngineCore                             ← v1/engine/core.py
   ├── Scheduler（唯一决策者）                  ← v1/core/sched/scheduler.py
   ├── Detokenizer / OutputProcessor
   └── Executor
        ├── UniProcExecutor（单卡）: ModelRunner 是【同进程对象】
        └── MultiprocExecutor（多卡）: spawn N 个 WorkerProc 子进程
             ↕ 共享内存 MessageQueue（不是 NCCL！）  ← shm_broadcast.py
             ├─ scheduler_output 广播: 1 写者（EngineCore）→ N 读者（worker）
             └─ worker_response: 每 worker 一个 MQ → driver worker 汇聚
```

关键事实（推翻我们 v6.4 的假设）：
1. **scheduler 与 executor 同进程**（EngineCore 内），worker 进程里没有 scheduler；
2. 多卡的调度下发走 **shm MessageQueue 广播**（pickle 的 Python 对象，结构化），
   不是固定形状 NCCL 张量——我们 v6.4 的固定张量方案是在"没有 EngineCore 主进程"
   的约束下发明的变通（N 个全等进程只能靠 NCCL 对话）；有了 EngineCore 形态，
   shm MQ 天然无错位问题（每条消息自带长度/元数据）；
3. **前端与 EngineCore 独立进程**（ZMQ）——流式输入输出解耦，engine 崩溃可重启；
4. NCCL 只用于 worker 之间的**前向内通信**（TP allreduce / EP a2a / PP p2p），
   不用于调度下发。

## 1. 我们的迁移映射

| vLLM 概念 | 我们的对应（现状 → 目标） |
|---|---|
| EngineCore（含 scheduler） | LLMEngine（现在含 scheduler+runner）→ 拆出 **EngineCore**（scheduler+executor） |
| WorkerProc（子进程） | 无 → **新增**：每 GPU 一个 mp.Process，内含 ModelRunner |
| MessageQueue（shm 广播） | v6.4 固定张量 NCCL broadcast → **替换为 shm MQ** |
| AsyncLLM + ZMQ 前端 | v5 的异步引擎 → 前端独立进程，ZMQ 连 EngineCore |
| UniProcExecutor | 单卡路径：worker 同进程直调（vLLM 同款退化） |
| virtual engine / micro-batch | 无 → PP 的流水调度 |
| data_parallel_rank 路由 | ParallelCoords 有 dp 维但无请求分发 → 实现 |

## 2. v9.4a：EngineCore + WorkerProc（本版核心）

```
v9/engine/core.py      EngineCore: busy loop = scheduler.schedule() → MQ 广播
                       → 等 driver worker 的 response → scheduler.update_from_output
v9/engine/worker.py    WorkerProc: 主循环 = MQ.recv(scheduler_output)
                       → model_runner.run() → response MQ.send
v9/engine/mq.py        轻量 shm 消息队列（multiprocessing.Queue 即可起步，
                       vLLM 的 shm_broadcast 是零拷贝优化，后续可选）
```

**SchedulerOutput 序列化**（对齐 vLLM 的结构化消息，不再是张量）：
```python
@dataclass
class SchedulerOutput:
    scheduled: list[SeqSpec]   # (seq_id, is_prefill, num_scheduled, block_table, tokens)
    finished: list[int]        # 本步完成的 seq_id
```
worker 收到后构造 forward 输入（v6.4 _SeqSpec 的职责，但每条消息独立、
无跨步状态 → 不存在"镜像状态列不全"的坑）。

**Worker 的 forward 输入同步**：TP/EP 下所有 worker 必须跑同一批 token。
vLLM 的保证 = scheduler_output 广播到所有 worker（同一条消息）——
我们的 MQ 广播同样保证。

**权重加载**：worker 进程内 ModelRunner 构造时读 ParallelState 切片
（v9.3 组件化的直接受益——launcher 只需在 fork 后 configure_parallel）。

## 3. v9.4b：DP

vLLM 语义：dp_size 个 **完整 EngineCore**（各自的 scheduler + KV cache +
完整模型副本），前端按 `data_parallel_rank` 路由请求（round-robin 或
load-aware）。**DP rank 之间零通信**（除了 metrics 汇聚）。

我们的实现：
```
前端进程: 请求队列 → dp_rank = i % dp_size 分发
EngineCore_i (i=0..dp_size-1): 独立 busy loop
   内部: TP/EP worker 们（DP 与 TP/EP 正交：dp×tp×ep 总进程数）
```
单机 8 卡的 DP2×TP4 = 2 个 EngineCore，各管 4 张卡。**TP 组在 EngineCore
内部的 worker 集合上建**（ ParallelCoords 的 (dp,tp,cp,ep,pp) 坐标直接适用）。

## 4. v9.4c：PP

vLLM 语义（源码要点）：
- `make_layers` + `get_pp_indices`: 每个阶段持层区间 [start, end)，
  区间外的层用 `PPMissingLayer`（Identity）占位——**层索引全局对齐**，
  checkpoint 名字不变；
- 微批（virtual engine）：EngineCore 维护 num_stages 个 in-flight batch，
  第 i 个微批的输出经过 p-1 次 p2p 激活传递后从最后 stage 流出；
- **scheduler 只在 stage 0 生效**（PP 的调度一致性由"同一个 scheduler_output
  广播到所有 stage 的 worker"保证——每 stage 收到同一批请求，但只算自己的层）。

我们的实现：
```
模型层: models/qwen3_moe.py 的 layers 用 get_pp_indices 切区间,
        区间外 PPMissingLayer 占位（loader 按名字跳过——param 不存在则跳）
worker: stage>0 的额外输入 = 上游传来的 hidden（p2p recv 替代 embedding）
        stage<p-1 的输出 = hidden 发给下游（不进 lm_head）
EngineCore: 微批调度（1F1B 起步，schedule 的 batch 池按 stage 数翻倍）
```

## 5. v9.4d：DCP 检查（回答"DCP 有没有集成在 layers/模型"）

**是的，已在组件层**：`layers/attention.py` 的 DCP 分支（store 门控 +
partial/LSE 合并）读 `get_dcp()` 的全局状态——worker 进程内 forward 时
生效，与 WorkerProc 架构天然兼容（每个 worker 独立持有自己的 cache 分片）。
**缺口**：v9 快照从 v8 拷来时 DCP 的 set_dcp 注入点要在 worker 初始化里
补一行（v8.2 是 launcher 注入的）。

## 6. 里程碑与验证

| 小版本 | 内容 | 验证（全部在 172.28.30.56 上跑） |
|---|---|---|
| v9.4a | EngineCore + WorkerProc + shm MQ | 单卡 inproc == 单进程逐位一致；TP2 双卡 token 级 vs 单卡 |
| v9.4b | DP2（×TP2 = 4 卡） | 两 dp rank 各自吞吐 ≈ 单 rank；输出与单卡一致 |
| v9.4c | PP2（微批 1F1B） | token 级 vs 单卡；bubble 实测 vs 理论 (p-1)/(m+p-1) |
| v9.4d | 全矩阵 DP×TP×EP×PP×DCP 扫描 | ParallelCoords 任意坐标可跑；对比表 |

## 7. 实验环境约束（用户指令）

**172.29.226.143（本机）禁止跑实验**——只做代码修改 + CPU 单测。
全部 GPU 实验迁 **172.28.30.56**（同机房，之前 v6 跨节点实测过的对端节点，
A800 ×8，SSH 免密已配）。跑前需确认：该机 Python 环境/模型缓存路径/
GPU 空闲状态。
