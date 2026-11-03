# v9 并行实现与 vLLM 的差异

## EP

旧实现把每个 rank 的 router 结果保留在本地，让每张卡对自己的专家区间计算，
最后在 `tp×ep` 组上 `all_reduce` 完整 hidden。它有两个问题：非本地 expert 的
路由信息仍在所有 rank 上重复处理；完整 hidden 在整个二维组内传输，通信量随 TP
和 EP 同时放大。

现在的 [moe.py](D:/vscode/vllm-learn/v9/layers/moe.py) 按 vLLM 的 dispatch/expert/combine
边界实现：

1. router 产生全局 expert id；
2. `all_to_all_single` 只把 token/expert pair 发到 owner EP rank；
3. owner 上执行本地专家 GEMM；
4. w2 的 TP partial 只在同一个 EP owner 的 TP group 内 all-reduce；
5. 反向 all-to-all 把结果按原 token id scatter 回去。

这使 EP 通信量近似与 `tokens × top_k × hidden` 成正比，而不是每层都对完整
`tokens × hidden` 在 `tp×ep` 大组上做 all-reduce。DeepEP/NCCL fused kernel 仍可在
后续替换 `_all_to_all`，而不需要改 router 或专家计算接口。

## TP 与 shared memory

vLLM 的 GPU TP 使用 NCCL；NCCL 在单机上会使用 NVLink/PCIe IPC 和 shared-memory
transport，跨机则使用 IB/网络。把 GPU tensor 先拷到 Python
`multiprocessing.shared_memory` 再做 TP all-reduce 会引入 GPU→CPU→GPU 往返，
性能和可扩展性都会明显变差，因此 v9 保留 GPU TP 的 NCCL collectives。

EngineCore 到 worker 的控制消息已经改为 [mq.py](D:/vscode/vllm-learn/v9/engine/mq.py)
的 shared-memory slot + notification/ack 协议：大对象只序列化一次，不再经过
Python Pipe 的 payload copy。它对应 vLLM 的 shared-memory message queue，属于
控制面；模型激活和 TP/EP 集合通信仍属于 GPU 数据面。

## DP/DDP

推理场景里的 DDP 实际语义是 data parallel replicas，而不是训练时梯度同步。
[v9/engine/frontend.py](D:/vscode/vllm-learn/v9/engine/frontend.py) 为每个 DP rank
创建独立 EngineCore，每个副本拥有自己的 scheduler、KV cache 和 TP/EP/PP worker
组；前端按 round-robin 或显式 `dp_rank` 路由请求，DP 副本之间不传激活，也不做
梯度 all-reduce。这与 vLLM 的 data-parallel serving 结构一致。

## 仍然不同的地方

- vLLM 使用 fused MoE kernels、拓扑感知的 all-to-all communicator 和 DeepEP；v9
  当前是 PyTorch `all_to_all_single` 加 Python metadata，正确性优先，kernel launch
  和 metadata overhead 更高。
- vLLM 的 scheduler/output metadata 是紧凑的 GPU/CPU结构，v9 仍是 pickle 后写入
  shared-memory slot，适合学习和验证，不等同于生产级 zero-copy tensor queue。
- v9 的 virtual-engine 是 microbatch slot 状态机；完整 vLLM 还会处理异步输出、抢占、
  speculative decoding 和 stale request 状态。
