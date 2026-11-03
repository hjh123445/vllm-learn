# Qwen3-Omni-30B 结构审计

## 结论

v9 的运行时入口收敛到 Qwen3-Omni-30B thinker。模型构造、权重装载和并行组件使用同一套接口；DP 是独立 EngineCore 进程，TP/EP/PP 在副本内部建立通信组。

## 高风险点

- Pipe 控制消息仍是 pickle，性能和故障恢复不如共享内存消息队列。
- PP 是 forward-only 双 lane 原型；还需要 CUDA 通信 stream 事件依赖、完整 stage 状态机、scheduler backpressure 和更细的 token chunking。
- EP 已按动态 `send_counts/recv_counts` 使用 all-to-allv，但专家 grouped GEMM 仍是 Python 循环，后续要做 persistent/fused kernel。
- trace 必须同时打开 CUDA/NCCL 活动并添加 layer、Attention、MoE、stage、microbatch、lane 标签，否则无法解释气泡。

## 验证顺序

先做单卡 BF16 正确性基线，再做 TP、EP、PP 双 lane、DP2×TP4；每次固定输入输出长度和并发，记录吞吐、TTFT、TPOT、显存、通信时间和专家负载方差。
