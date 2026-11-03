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

EngineCore 在一个调度步内按 token 预算把多条序列分成微批，每步最多让两个消息在途；`PipelineTransport` 在worker进入recv后懒投递shape `irecv`，知道shape后再分配payload，发送端用`isend`和credit保留buffer。worker每次仍处理一个scheduler消息，接收阶段仍等待；单条sequence不在microbatch内部切分，超过token预算会单独形成超预算消息。

当前 runtime 是带 microbatch/lane 的 forward-only PP prototype，不是 DualPipe，也不保证 zero-bubble。`build_forward_schedule` 只生成理想化测试计划，不能当作实际 worker schedule。单个长请求在消息内保持完整，chunked prefill 跨调度步推进；微批不足时仍有 fill/drain bubble。实际重叠需查看同一时间基准下的 CUDA stream trace。

## EP

Qwen3-Omni 的专家 token 数动态变化，每一步先交换 `send_counts/recv_counts`，再使用 all-to-allv dispatch/combine；不会用固定 `[world, max_tokens]` padding。后续 kernel 路线是 fused router+top-k、persistent grouped GEMM、dispatch/combine 融合和通信计算 overlap。

## 实验门槛

每个配置都要通过单卡数值基线、TP/EP/PP 端到端检查、变长计数检查、异常退出回收和 `torch.profiler` trace。trace 至少包含 CUDA kernel、NCCL/P2P、层号、Attention/MoE 标签、PP stage、microbatch/lane 和请求时间线。
