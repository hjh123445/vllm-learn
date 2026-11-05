# 当前并行实现与 vLLM 的差异（2026-11-04）

对照官方 commit `d6b48f928f922849db55751ceadbbb55a5a85260`；实际待测官方基线为隔离的 0.13.0。版本不同，必须分别标注。本文描述实现边界，不代表所有功能通过验收。

## 当前EP拓扑已改为DP×TP强耦合

2026-11-05起设备数为DP×PP×TP×CP，EP不额外乘卡。启用EP时，固定PP/CP坐标下EP组覆盖DP×TP。Attention在DP副本内做TP；MoE持E/(DP×TP)个完整专家，关闭EP才保留专家intermediate TP。旧独立TP×EP数据只作为历史证据，不能拿它代表新结构。

数据路径：router完整DP输入 → 按TP rank划分token ownership → 跨DP×TP变长all-to-allv → 本地完整专家 → reverse combine → 聚合top-k → 只在原DP副本TP组内gather恢复。TP边界padding最多TP−1行，不是专家容量padding。不存在旧版combine之后的专家TP partial all-reduce。

各DP输入长度/内容可以不同，空DP仍必须参与MoE collective。DPFrontend为每副本启动独立EngineCore，共享一个EP通信world，每轮统一驱动。仍有CPU join屏障；不支持coupled DP的per_request执行，也不等于官方完整异步调度、DeepEP或EPLB。详见EP_DP_TP_IMPLEMENTATION.md。

## 为什么尚不等于高效 fused MoE / DeepEP

本项目已有 dispatch/expert/combine 边界，但 _experts_local 仍逐专家执行 PyTorch GEMM，counts.tolist() 要把设备结果交给 Python，dispatch metadata 也分多次 collective。当前 backend 没有 DeepEP 的专用 dispatch/combine 内核、低延迟协议或细粒度计算通信 overlap。函数边界可替换不等于简单换一个 all_to_all 调用就具备 DeepEP。

下一步先以同输入 route oracle 验证数值，再用 profiler 确定 metadata、launch、GEMM、NCCL 等占比。按收益评估 GPU route layout/offset、grouped GEMM、activation/weighted combine 融合；DeepEP 后端需要对应协议、buffer 生命周期、拓扑支持与调度配合。Qwen3-Omni 此文本 MoE 没有共享专家，不能直接套用“共享专家掩盖 all-to-all”实验。

## TP 与 shared memory

CPU 控制消息已经走 shared-memory slot + notification/ACK；GPU TP 目前保留 NCCL。把 GPU tensor 放进 Python multiprocessing.shared_memory 会增加 D2H/H2D。官方还可按设备/消息条件使用 CUDA IPC custom all-reduce 等；这属于 GPU 可访问内存与专用归约协议，不能与 Python CPU shared memory 混为一谈。

## PP、DP 与验收范围

PP 在 2026-11-05 改为单 scheduler + 跨 step 持久 batch_queue/Future，默认并发批数等于 PP size；独立的每 worker 提交线程和结果接收线程解除控制面 ACK 屏障。prompt chunk 与 decode 请求在 scheduler 内形成有界批次；在途请求/KV 固定到结果回收，取消延迟释放。模型返回 IntermediateTensors，runner 管理 P2P。邻接 P2P communicator 默认启用，仍需匹配负载的 NCCL 和吞吐验收；接收仍有 header readback，未移植官方全部 lazy recv/异步 token feedback。没有训练 backward、完整 DualPipe/zero-bubble、同卡多 stage。见 PP_V1_IMPLEMENTATION.md。

DP 每副本独立 EngineCore 与 GPU 集合，语义是推理 replica。前端每 step join，仍有慢副本屏障。官方跨 DP 的 EP 要保证 collective 对齐与空 batch 协调，本项目已增加该协调，但仍为同步轮次，不能概括成官方完整异步实现。

完整模型精度、官方同机基线、真实多模态 serving、量化 serving 与投机解码均需单独验收。实验记录必须注明源码指纹、模型、版本、输入/输出长度、dtype、计时边界、采样覆盖和失败情况。
