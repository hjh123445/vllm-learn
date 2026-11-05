# v9 当前进程与并行架构（2026-11-04）

本文描述当前代码，并把已实现、参考设计和未完成能力分开。支持拓扑参数不代表任意组合均通过模型精度与性能验收。实验、原始数据和中文分析见本任务 outputs/verified_parallel。

## 参考版本

源码对照为官方 vLLM commit `d6b48f928f922849db55751ceadbbb55a5a85260`，远端只读路径 `/data/ryxu/vllm`，版本描述 `v0.21.1rc0-359-gd6b48f928`。准备运行的隔离基线是 vLLM **0.13.0**，两者必须分开标注。API、executor 和微批调度随版本变化，本文不把某一版的实现当作所有 vLLM 版本的规则。

## 进程和控制面

```text
DPFrontend（离线前端、请求全局 ID、round-robin 路由）
  ├─ DP replica 0：独立 supervisor / EngineCore
  │    ├─ Scheduler、KV block 元数据、Executor 驱动
  │    └─ 每 GPU 一个 WorkerProc / ModelRunner
  └─ DP replica 1：另一组 supervisor / EngineCore / WorkerProc
       ...
```

本项目 engine/frontend.py 为各 DP 副本创建独立进程树和互斥 GPU 集合；每副本各有 scheduler、模型权重和 KV cache。step_all() 用线程并发驱动各副本再 join，仍有慢副本屏障；它不是完全异步在线服务前端，也不是训练 DDP。

engine/core.py 的 EngineCore 是本副本 scheduler 的唯一 owner，生成结构化 step/microbatch 消息。engine/mq.py 使用 pickle、共享内存槽、notification 与 ACK；这是控制面，不是 GPU tensor 的零拷贝传输。worker 的 TP/EP/PP 激活通信在 GPU 数据面执行。

官方 v1 通常把 scheduler 和 executor 放在 EngineCore；前端/EngineCore 由客户端协议连接，多进程 executor 可使用共享内存消息队列。detokenizer/output processor 通常属于前端输出处理链，不应画成 EngineCore 固有成员。具体单/多进程方式取决于 executor 和版本。

## 本项目拓扑不等于官方 vLLM 拓扑

本项目物理 worker 数为 `DP × PP × EP × TP × CP`。TP、EP 是正交坐标；EP 切专家编号，TP 还会切同一个专家的 intermediate 维。DCP 也使用本项目独立的 CP 坐标，不能直接套用官方 DCP 的 GPU 数公式。

官方参考版本 vllm/config/parallel.py:152 将 EP 描述为 MoE 层采用专家并行来代替 tensor parallel。parallel.py:769 中单 DP 副本 world size 为 PP×TP×prefill-CP；distributed/parallel_state.py:1688 的 EP group 大小包含 DP×prefill-CP×TP。EP 通常复用 attention 的 TP/DP ranks，并不是额外乘出的设备轴；官方 DCP 同样复用 TP 设备（parallel_state.py:1611）。

因此，本项目 **DP2×EP4 是两个各自完整的 EP4 专家池**。官方 attention-DP2 与跨 DP 的 EP 组合可以让两个 attention 分片共享更大的专家池，必须协调 collective/空 batch，不能说 DP ranks 永远零通信。官方 TP8+EP 在 8 张卡上采用 attention-TP8 / expert-EP8；本项目 TP8×EP8 则需要 64 张卡，是另一种结构。

本项目当前 EP peers 收到同一 attention batch，默认重复 dense attention/router。opt-in ownership 仅切分进入 MoE 的 token，combine 后 gather 恢复复制 hidden；它不等于完整 attention-DP。官方 EP 也不能概括为所有 ranks 都处理相同 token。

## TP、EP 和装载边界

layers/linear.py、layers/parallel.py、layers/moe.py 的组件持有 shard 与 collective 语义；loader 做 checkpoint 名称映射，再调用组件 weight_loader。当前不是在额外 parallel.py 中通过 monkey patch 劫持整个模型 forward。

本地专家权重为 `[E_local, 2*I_local, H]` 与 `[E_local, H, I_local]`。EP dispatch 交换 per-peer counts 后使用变长 all_to_all_single，expert 计算后 reverse combine；源 rank 恢复 route 顺序、聚合 top-k，再在 TP group 归约专家 intermediate partial。该 TP 归约是本项目布局的要求，不可原样解释官方 EP 内核。

模型激活仍用 NCCL；shared-memory MQ 解决 CPU 调度消息广播。官方 GPU communicator 还可能按硬件和消息条件选 CUDA IPC custom all-reduce 等，不能说 vLLM 的 TP 一律只走 NCCL，更不能把 Python shared memory 当成 NVLink 通信替代。

## PP 和 virtual engine 的准确状态

PP 按全局 layer index 连续切分，非本 stage 的层用不持参数的占位模块保持 checkpoint 名字。PP2 对 48 层为 L0–23 / L24–47；stage 间传 hidden 与 residual。

当前 EngineCore 按完整 Sequence 组成 microbatch，不能在同一个 step 内任意切开一个长 Sequence；token budget 对单个超长请求不是硬边界。VirtualEnginePool 管理在途 microbatch 的 lane/slot 状态，Core 窗口最多两条消息。worker 一次处理一条消息，接收 header 后分配 payload 并等待；send 可在独立 communication stream 异步进行，有 CUDA event 和 send credit。可选邻接 stage 专用 P2P group 正在与 world group 对照。

这些是 forward-only 推理微批原型。没有 backward，因此不能叫 1F1B；lane 不代表同卡多 stage。非一一 stage 映射目前明确拒绝，完整 virtual-engine scheduler、同卡多 chunk 交错、预投递接收、DeepSeek 训练 DualPipe 和 zero-bubble 均未完成。scheduler 位于 EngineCore host，不是“stage 0 的 GPU 内”。

数值复核可通过 `batch_execution_mode="per_request"` 让每个 PP worker 按同一顺序逐条执行 batch。这个保守模式已在 PP2×EP4 短输入小样本验证 stage0/stage1 边界，仍会牺牲批量融合；默认 `batched` 路径的跨 M 形状 BF16 数值差异仍需用真实负载验收。

## 验收边界与当前环境

用户已授权在 172.29.226.143 的任务仓库进行 8×A800 实验，2026-11-04 北京时间 09:00 前停止，本任务 supervisor 截止设为 08:59。只管理本任务 process group。

CPU 拓扑/状态机、四卡 TP2×EP2 的小张量 oracle、若干模型拓扑运行与 greedy 重复检查已经提供证据，但**全模型 HF 数值认证尚未通过，跨拓扑 greedy 有分歧**。因此性能表只表示当前实现的测量，不能据此宣称与官方等价、正确性全面通过或优于 vLLM。多模态 HF probe、量化组件和投机解码历史原型也不能当作当前 serving 已集成能力。
