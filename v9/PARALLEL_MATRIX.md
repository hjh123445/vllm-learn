# Qwen3-Omni-30B 并行实验矩阵

本文只讨论 Qwen3-Omni-30B-A3B thinker。所有速度数字都必须带上模型 commit、GPU、请求数、输入长度、输出长度和精度，不能混用不同环境的数字。

## 并行语义

| 维度 | 切分对象 | 进程关系 | 主要通信 |
|---|---|---|---|
| TP | 每层线性层的输入/输出维 | 一个 EngineCore 内的 worker | RowParallel all-reduce |
| EP | MoE 专家和 token 路由 | 一个 EngineCore 内的 worker | dispatch/combine all-to-allv |
| PP | 连续 decoder layer | 一个 EngineCore 内的 stage worker | 相邻 stage P2P activation |
| DP | 完整模型副本和请求/KV 池 | 多个独立 EngineCore 进程 | 前端路由，无 NCCL |

DP2×TP4 的拓扑是两个独立进程：DP0 管 GPU 0--3，DP1 管 GPU 4--7。每个进程各自拥有 scheduler、KV cache、TP/EP/PP 通信组和 worker；请求进入某个副本后保持粘性。`v9/engine/frontend.py` 用控制 Pipe 启动这些 EngineCore，物理 GPU 列表只在子进程中转成局部编号，避免两个副本共享状态。

## PP 与 DualPipe

旧路径是一次 scheduler 输出对应一次完整 forward：stage 0 算完后发送，stage 1 阻塞接收，最后一个 stage 返回结果。现在 EngineCore 在一个调度步内按 token 预算切成微批，先下发所有微批再收集结果；`PipelineTransport` 为每个 lane 预投递 shape `irecv`，payload 用 `irecv` 接收，发送端使用 `isend` 和 credit 窗口保留 buffer。每条消息带 `microbatch_id`、`pipeline_lane` 和 shape，结果按 `seq_id` 重排。

这仍是 forward-only DualPipe，没有 backward、梯度、参数更新或训练版 1F1B 反向优先级调度。单个长请求仍在一个消息中保持完整，chunked prefill 只让它跨多个调度步推进；微批不足时 fill/drain bubble 仍存在。理想 `m` 个微批、`p` 个 stage 的 bubble 约为 `(p-1)/(m+p-1)`，必须用 CUDA trace 验证真实重叠后才能谈 zero-bubble。

## EP

Qwen3-Omni 的专家 token 数动态变化，每一步先交换 `send_counts/recv_counts`，再使用 all-to-allv dispatch/combine；不会用固定 `[world, max_tokens]` padding。后续 kernel 路线是 fused router+top-k、persistent grouped GEMM、dispatch/combine 融合和通信计算 overlap。

## 实验门槛

每个配置都要通过单卡数值基线、TP/EP/PP 端到端检查、变长计数检查、异常退出回收和 `torch.profiler` trace。trace 至少包含 CUDA kernel、NCCL/P2P、层号、Attention/MoE 标签、PP stage、microbatch/lane 和请求时间线。
