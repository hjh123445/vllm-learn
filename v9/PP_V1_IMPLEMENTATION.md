> 2026-11-05 EP拓扑更新：EP不再是独立设备轴；DP×TP派生EP、空副本参与与新接口见 [EP_DP_TP_IMPLEMENTATION.md](EP_DP_TP_IMPLEMENTATION.md)。下文旧TP×EP验证数据仅为当时快照。

# V1 风格推理流水线实现说明

日期：2026-11-05。参考 vLLM v0.31.0 的普通 batch queue 路径；本文描述本次修改，不把历史吞吐和 trace 当作新实现的结果。

## 解决的问题

旧 EngineCore 每次 schedule 后才切微批，最多提交两条，必须收齐本轮结果再 postprocess。两个限制叠加：新请求/已完成微批的下一 token 要等尾批；decode token 很少时又可能全部装入一个微批。VirtualEnginePool 只做 mb_id % lanes 计数，不具备独立 scheduler/KV 或实际执行能力。

控制面也有限制。原 broadcast 向所有 worker 写入一次消息后，等待所有 worker 的读取 ACK，才允许发下一条。末 stage 还在处理早期消息时，会拖住首 stage 后续消息。输出也只有一个共享槽，若 Core 等控制 ACK、末 worker 等输出 ACK，可形成环路。仅把窗口从 2 改为 4 并不能正确解决。

## 新的运行主链路

```text
AsyncEngine / generate / DPFrontend
             ↓ 单线程拥有调度状态
EngineCore.step
  1. 队列有空间：scheduler.schedule 形成一个有界 batch
  2. executor.execute_model 立即返回 Future
  3. 保存 PendingBatch，未满则返回，不等待 GPU
  4. 满/无可调度请求：只等队首 Future，更新该批请求
  5. 尾批留在队列，下一次 step 可补入新的工作
             ↓
MultiprocessExecutor
  每个 worker 独立 FIFO 提交线程 → SHM input slot → worker
  独立结果线程 ← SHM output slot ← 末 stage 的唯一 output rank
             ↓
worker_main → ModelRunner.run
  首 stage：embedding → 本地层 → IntermediateTensors
  中间 stage：recv → 本地层 → IntermediateTensors → send
  末 stage：recv → 本地层 → final norm/head/sample → response
```

默认 `max_concurrent_batches=pp_size`；可以显式设 1 做串行提交消融。队列容量约束所有尚未回收的 batch，已完成但未回收的 Future 也计入；不以 CUDA 是否已经执行完来放松内存/请求所有权。

提交函数不会等待 worker ACK 或 GPU 完成。每个发送线程只操作自己的 SHM slot，下一次写入前等该 worker 已拷贝输入的 ACK。不同 rank 无全局 ACK 屏障，但同一 rank 的消息严格有序。独立接收线程读完结果即 ACK，然后完成 Future，不必等 scheduler 消费结果。后台线程不访问 Sequence、scheduler 或 BlockManager。

## 代码职责

| 文件 | 关键对象 | 职责 |
|---|---|---|
| engine/core.py | EngineCore、_PendingBatch | 持久 batch_queue、输入快照、提交/按序回收、API 和退出 |
| engine/executor.py | MultiprocessExecutor | 非阻塞控制面 RPC、每 rank FIFO、独立接收、健康检查和错误传播 |
| engine/scheduler.py | Scheduler | 在途请求保护、token/request budget、取消延迟释放、结果结算 |
| engine/intermediate_tensors.py | IntermediateTensors | 明确的 `[T,H]` hidden/residual 模型边界 |
| engine/model_runner.py | ModelRunner | 输入准备、PP 收发、本地模型调用、末 stage 采样 |
| models/qwen3_moe.py | Qwen3MoeModel | 仅计算 `pp_start:pp_end` 的层；不再直接调用 transport |
| engine/pipeline_transport.py | PipelineTransport | NCCL P2P、stream/event、发送引用存活与本地 buffer credit |
| engine/worker.py | worker_main | 设备和组初始化、消息执行、唯一结果 rank、profile/退出 |

`VirtualEnginePool` 已删除。`pipeline_lanes` 参数和 transport 的 lane 字段为旧测试/调用兼容保留；新 Core 固定使用 slot 0。并发度由 batch_queue 决定，不能通过增加 lanes 增加 scheduler 或 GPU stage。

## 微批与自回归依赖

PP 时 scheduler 直接使用 `min(pipeline_microbatch_tokens, max_num_batched_tokens)` 作为每批硬 token 预算。一个长 prompt 可以形成多个 chunk，不会因完整 Sequence 大于预算就产生超大微批。

每批请求上限取 `ceil(min(活跃请求数,max_num_seqs)/max_concurrent_batches)`，最小 1，使短 decode 也能形成多个 batch。这是本项目的简单动态分批策略，不是复制官方所有调度启发式；需实验选择合适的预算和并发深度。

每个请求最多一个在途 batch。只有对应结果回收后，才能继续安排该请求的下一个 chunk/token。这里没有 output placeholders，也不预排尚未得到 token 值的同请求下一次 decode。因此这是 V1 普通多批并行主线，不是完整 async scheduling 的 token feedback 实现。

只有一个短请求 decode 时仍有 PP 气泡；队列槽位不会创造额外独立工作。同一长 prompt 的 chunks 也采用保守的结果回收依赖；没有把单请求多个 chunks 同时放入不同 stage。

## 请求和 KV 的正确性约束

1. schedule 立即把 seq_id 放入 `in_flight`，保留 num_scheduled_tokens 与 block_table；后续调度跳过它，不重复改写输入范围。
2. 抢占 victim 排除之前批次的 in_flight 请求，以及当前批已经选中的请求。
3. abort 对在途请求只记录取消，继续消费其输出，不输出 token。直到匹配结果到达才释放 block。
4. 未完成 GPU 的新 KV 不发布到 prefix hash；只在结果结算时 hash_blocks。
5. executor 检查 output type、batch ID、完整有序 seq_ids 和 token 数；Core 再检查其队首归属。错误直接失败，不静默用 zip 截断。
6. P2P receive 检查 batch ID、lane 和预期 `[T,H]`；接收 dtype 由同一模型配置传入，不再强行使用 BF16。旧四整数 header 本身不协商 dtype，要求各 worker 配置一致。
7. 输入为独立 Python 列表/字典快照；后续 Sequence 更新不影响已经提交的消息。KV block size 一并传递，修复 512 等配置被 worker 固定按 256 解释的问题。
8. shutdown 停止新提交，取消等待请求、回收所有在途结果，再发送 shutdown。发生 worker 错误/超时只终止本 Core 的子进程，停止 IPC 线程后关闭共享槽。

`num_cached_tokens` 在本实现中仍是已经结算的计数；尚在途的范围由 `in_flight + num_scheduled_tokens` 单独表示，不把已安排和已完成混作一个变量。

## GPU 通信与 buffer

主要 activation 仍在 GPU，TP/EP/PP 使用 NCCL。PP 默认使用邻接 stage 的专用 P2P communicator；可设 `VLLM_LEARN_PP_P2P_GROUPS=world` 做对照。通信 stream 等待生产 activation 的 compute event，发送记录持有 header/hidden/residual 引用，达到本地 send credit 后回收最早发送。

模型边界同时传 hidden 与 residual，因为 fused_add_rmsnorm 延迟了残差相加。不能只传 hidden。模型不再负责收发，也不对区间外 PPMissingLayer 生成假的 layer trace。

当前 receive 仍先读 header 到 CPU，再分配/接收 payload；没有移植官方 AsyncIntermediateTensors 的 lazy accessor。独立 comm stream、非阻塞 send 和多个 Future 都不自动证明实际计算通信重叠。需要真实 CUDA/NCCL trace 和无 profiler 吞吐对照确认。

新增持久发送 buffer 会占用显存；send credit 是每个 transport 的本地回收边界，不是远端 credit 协议。30B 压力场景下仍需测激活峰值和 KV 安全余量。

## 使用

```python
core = EngineCore(
    model_path,
    pp_size=4, ep_size=1, tp_size=1,
    pipeline_microbatch_tokens=512,
    max_concurrent_batches=4,  # 默认就是 pp_size
)
```

HTTP 服务支持 `--max-concurrent-batches`，AsyncEngine 和 DPFrontend 也沿用同一 Core 路径。`step()` 可能只提交而返回空列表；用 `is_finished()` 判断是否结束，不要把一次无输出当作无在途工作。

## 验证方式与证据

本次结果保存在 `outputs/verified_parallel/pp_v1_20261105/`；CPU/NCCL 子目录均保存测试日志、JUnit XML、执行命令、状态和源码 SHA256 manifest。远程使用 `/tmp/vllm-learn-pp-v1-*` 独立源码快照，不覆盖远程工作仓库，不推送 GitHub。

- test_pipeline_batch_queue.py：真实 scheduler/BlockManager 配合可控 Future，验证跨 step 持续补队、chunk/decode 串行一致、KV 压力、在途取消、prefix 发布、错误结果、shutdown。
- test_pipeline_executor.py：真实 spawn 和 SHM IPC；故意阻止末 stage 读首条控制消息，检查首 stage 已能读到第 4 条；8 批在途验证输出 ACK 独立回收；注入错 batch/worker 退出。
- test_pipeline_model_runtime.py：小型随机 FP32 Qwen MoE、真实 ModelRunner/Scheduler/Executor；CPU/Gloo 和 GPU/NCCL 下 PP4、PP2×EP2、PP2×TP2 与串行缓存生成参考比较。
- test_pipeline_worker_runtime.py：显式 NCCL 模式；临时生成小型 BF16 checkpoint/tokenizer，直接启动生产 EngineCore/worker_main，覆盖真实 loader、warmup、专用 PP groups、混合并行、生成、在途取消和退出。

这些结果不等于 Qwen3-Omni-30B 全模型 HF 认证，不证明 BF16 跨不同 GEMM 形状严格一致，也不提供优化前后 tok/s。完整 checkpoint 的数值、吞吐和长尾负载实验需单独测量；没有新数据前不复用旧报告中的性能数字。

## 官方参考

- vLLM v0.31.0 `vllm/v1/engine/core.py`：batch_queue、step_with_batch_queue。
- `vllm/config/vllm.py`：max_concurrent_batches。
- `vllm/v1/executor/multiproc_executor.py`：non_block collective_rpc。
- `vllm/v1/worker/gpu_worker.py`：AsyncIntermediateTensors、irecv/isend、发送 buffer 存活。
- `vllm/model_executor/models/qwen3_moe.py`：local layer range 与 IntermediateTensors。

本项目没有 backward、interleaved 多层 chunk 或双向流水线，不能将这次修改称为训练 DualPipe、1F1B 或 zero-bubble。
