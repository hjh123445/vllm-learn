# Qwen3-Omni-30B PP/DualPipe 说明

## 一次完整 forward 为什么有气泡

假设有 p 个 stage 和一个 batch。stage0 必须先算完并发送 activation，stage1 才能开始；stage1 工作时 stage0 等待下一批，最后还要排空后续 stage。这个等待区间就是 pipeline bubble。m 个独立微批进入后，stage0 可以继续计算下一个微批，stage1 同时处理上一个微批，理想 bubble 约为 (p-1)/(m+p-1)。

## 当前代码做了什么

EngineCore 把 scheduler 的多个序列按 token 预算切成微批，为每个微批分配单调递增的 microbatch_id 和两个 lane 之一；每步最多允许两个 microbatch 在途，达到窗口后就从最后 stage 收集结果。单条序列不会在微批内部拆开，因此序列长度大于 token 预算时仍会单独形成超预算消息；chunked prefill 负责让它跨调度步推进。

PipelineTransport 在 worker 进入 recv 消息时懒投递 header irecv，收到 header 后才分配 payload 并投递 irecv；stage<last 使用异步 isend。操作使用独立通信 stream，并用 CUDA event 与计算 stream 建立依赖，credit 限制每 lane 的未完成发送数。发送 buffer 在对应 Work 完成前保持引用，worker 退出前 drain 未完成发送。当前没有预先投递下一条 header receive。

因此 stage0 的后续微批可以在 stage1 计算前一个微批时开始异步发送，但 worker 仍按消息逐条处理。这是推理 forward-only 的双 lane PP prototype，不包含训练 DualPipe 的 backward 调度，也没有证明 zero-bubble。

## 仍然存在的边界

单个超长请求在一个调度步内仍按完整序列传输；chunked prefill 会在多个调度步推进它，但不能只切 activation 而不同时切 KV 写入。请求数少于流水线深度时仍会有 fill/drain bubble。要声称 zero-bubble，必须在 CUDA trace 中同时看到计算 stream 和通信 stream 的重叠，并量化空闲比例。

header 仍然需要先完成，才能按实际 token 数分配 hidden/residual；因此这是异步 payload，而不是完全固定 buffer 的预投递 all-to-all。worker 的 Python 主循环仍一次处理一条 scheduler 消息，通信与相邻 stage 的计算可以重叠，但同一个 stage 没有多个独立计算线程。

## 面试时应如何表述

可以说项目实现了“动态 token-budget microbatch、双 lane 元数据、异步 P2P、通信 stream、CUDA event 和有限 send credit 的 forward-only pipeline prototype”，并明确还没有实现训练版 DualPipe 的 backward 优先调度，也没有 DeepEP 级别的自定义通信 kernel。应以实际 CUDA trace 说明是否存在有效重叠。
