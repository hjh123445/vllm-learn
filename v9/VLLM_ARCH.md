# v9 当前进程与并行架构（2026-11-05）

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

本项目 engine/frontend.py 为各 DP 副本创建独立进程树和互斥 GPU 集合；每副本各有 scheduler 和 KV cache；开启EP后专家权重在副本之间分片共享。step_all() 用线程并发驱动各副本再 join，仍有慢副本屏障；它不是完全异步在线服务前端，也不是训练 DDP。

engine/core.py 的 EngineCore 是本副本 scheduler 的唯一 owner，生成结构化 step/microbatch 消息。engine/mq.py 使用 pickle、共享内存槽、notification 与 ACK；这是控制面，不是 GPU tensor 的零拷贝传输。worker 的 TP/EP/PP 激活通信在 GPU 数据面执行。

官方 v1 通常把 scheduler 和 executor 放在 EngineCore；前端/EngineCore 由客户端协议连接，多进程 executor 可使用共享内存消息队列。detokenizer/output processor 通常属于前端输出处理链，不应画成 EngineCore 固有成员。具体单/多进程方式取决于 executor 和版本。

## 当前拓扑：EP 复用 DP×TP

2026-11-05 按官方思路取消独立EP设备轴。worker总数为 `DP × PP × TP × CP`；开启 `enable_expert_parallel=True` 后，每个固定PP/CP坐标的EP组包含全部DP×TP ranks，`ep_rank=dp_rank*tp_size+tp_rank`。每个DP副本有自己的EngineCore、scheduler、KV；Attention仅在该副本的TP组内切头和归约。MoE专家沿EP组分配，每个owner持完整专家中间维，不再对同一专家做TP切分。

本项目CP仍是独立设备维，与官方复用TP ranks的DCP不同。本文的EP=DP×TP适用于没有prefill-CP的范围；官方包含PCP等配置时组大小需以对应版本为准。

DP启用EP后共享专家池，需要统一collective轮次。离线DPFrontend每轮驱动所有副本；空闲副本执行本stage所有MoE空输入，仍接收并计算其他DP来源的专家路由。跨DP每轮join是简化实现，尚未达到官方异步负载协调；coupled DP模式拒绝per_request，避免副本请求数不同而执行不同次数collective。详细接口、拓扑与验收见 EP_DP_TP_IMPLEMENTATION.md。

## TP、EP 和装载边界

layers/linear.py、layers/parallel.py、layers/moe.py 的组件持有 shard 与 collective 语义；loader 做 checkpoint 名称映射，再调用组件 weight_loader。当前不是在额外 parallel.py 中通过 monkey patch 劫持整个模型 forward。

本地专家权重为 `[E_local, 2*I_local, H]` 与 `[E_local, H, I_local]`。EP dispatch 交换 per-peer counts 后使用变长 all_to_all_single，expert 计算后 reverse combine；源 rank 恢复 route 顺序、聚合 top-k，最后只在原DP副本的TP组内gather恢复token顺序。EP开启时没有专家TP部分和，不再追加TP all-reduce。

模型激活仍用 NCCL；shared-memory MQ 解决 CPU 调度消息广播。官方 GPU communicator 还可能按硬件和消息条件选 CUDA IPC custom all-reduce 等，不能说 vLLM 的 TP 一律只走 NCCL，更不能把 Python shared memory 当成 NVLink 通信替代。

## PP 和 virtual engine 的准确状态

PP 按全局 layer index 连续切分，非本 stage 的层用不持参数的占位模块保持 checkpoint 名字。PP2 对 48 层为 L0–23 / L24–47；stage 间传 hidden 与 residual。

2026-11-05 改为 V1 风格单 scheduler + 持久化 batch_queue + Future。默认 max_concurrent_batches=PP size；队列未满优先提交，没有可调度请求或队列满时只回收最早一批，未完成尾批保留到下一次 Core 调用。scheduler.schedule 直接按 pipeline_microbatch_tokens 切 prompt chunk，并限制每批请求数使 decode 也能分批；每个请求最多一个在途 batch，取消后延迟释放 KV，抢占排除在途请求。

executor 为每个 worker 提供独立提交线程和有序消息队列，独立结果线程立即读取并 ACK 输出槽，解除原先两批上限和跨 stage ACK 屏障。模型仅接收/返回 IntermediateTensors；runner 管理实际 PP 收发，检查 batch ID 和 shape。默认使用邻接 stage 专用 P2P group；原始 world 选项保留用于消融。VirtualEnginePool 已删除；兼容参数 lane 仅是 transport 标签，运行时固定使用 slot 0，不决定并发度。

这是 forward-only inference PP，对齐 V1 普通 batch-queue 主线；尚未实现 output placeholder 异步 token 反馈、同请求多批预排、接收 tensor 的 lazy accessor、同卡多 stage、训练 DualPipe/1F1B/zero-bubble。接收仍有 header CPU readback，独立通信 stream 不自动保证真实重叠。详见 PP_V1_IMPLEMENTATION.md。

数值复核可通过 `batch_execution_mode="per_request"` 让每个 PP worker 按同一顺序逐条执行 batch。这个保守模式已在 PP2×EP4 短输入小样本验证 stage0/stage1 边界，仍会牺牲批量融合；默认 `batched` 路径的跨 M 形状 BF16 数值差异仍需用真实负载验收。

## 验收边界与当前环境

用户已授权在 172.29.226.143 的任务仓库进行 8×A800 实验，2026-11-04 北京时间 09:00 前停止，本任务 supervisor 截止设为 08:59。只管理本任务 process group。

CPU 拓扑/状态机、四卡 TP2×EP2 的小张量 oracle、若干模型拓扑运行与 greedy 重复检查已经提供证据，但**全模型 HF 数值认证尚未通过，跨拓扑 greedy 有分歧**。因此性能表只表示当前实现的测量，不能据此宣称与官方等价、正确性全面通过或优于 vLLM。多模态 HF probe、量化组件和投机解码历史原型也不能当作当前 serving 已集成能力。
