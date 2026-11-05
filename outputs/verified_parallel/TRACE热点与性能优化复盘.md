# Trace 热点与性能优化复盘

## 这份复盘回答什么问题

这份复盘把性能分析分成三层：先确认 trace 是否完整，再从时间线定位热点和气泡，最后用同口径吞吐、尾延迟和正确性回归判断修改是否真的有效。当前项目已经有一个确定的系统优化成果，也完成了几项 MoE/PP 优化原型；但 trace 还没有证明项目已经达到 zero-bubble 或超过 vLLM，因此报告把已验收结果、候选收益和负结果分开记录。

## 已经确认的优化成果

### 1. 初始化阶段的 CPU 线程过度订阅

早期 DP2×EP4 启动和首次 warmup 约 410–413 秒。每个 worker/core 都继承较大的 OMP/MKL 线程池后，多个进程在同一台机器上争抢 CPU，GPU 尚未成为主要瓶颈。将线程限制同步到 worker/core（默认 `VLLM_LEARN_CPU_THREADS=4`，并在启动时补齐 OMP/MKL/PyTorch 线程设置）后：

| 配置 | 初始化加首次 warmup |
|---|---:|
| 未限制线程 | 约 410–413 s |
| 显式线程数 4 | 约 34.65 s |
| 默认线程控制复核 | 34.55 s |

这相当于约 **11.9×** 的启动阶段加速。它不应写成 decode tok/s 加速，也不应外推到所有 CPU 核数；不同 NUMA、worker 数和权重存储仍需重新调参。证据：`dp2ep4_default_threads/` 与 `dp2ep4_default_threads.log`。

### 2. MoE grouped backend 原型

trace 显示专家执行包含大量小矩阵乘法和 Python expert loop。项目增加了 opt-in `torch_grouped` 后端，把同一层的多专家计算改为 grouped GEMM 形式，减少逐专家 Python 调度。早期 EP8 矩阵中，loop-refactored/旧数学路径相对基线在六类负载上约有 **3.7%–9.8%** 的局部提升。

这组结果只能说明 grouped 计算值得继续做，不能当成当前生产收益：它使用的是较早的数学路径，尚未通过最终整模数值门槛；同时 `torch_grouped` 仍不是融合 dispatch、expert GEMM、combine 的 CUDA kernel。面试时应说“完成了可回退的 grouped 原型并观察到局部收益，随后因为精度和端到端口径没有把它默认打开”。

### 3. PP 异步传输原型暴露了真正的限制

PP 路径增加了非阻塞 send、专用通信 stream 和有限 credit。trace 中两 stage 的普通计算区间交集从 baseline 约 **20.06%**，到 optimized 短窗口约 **22.39%**，dedicated 窗口约 **25.05%**。这些是“采样窗口内 stage 计算时间的区间交集”，不是 SM 利用率，也不是稳态 bubble 比例。

同口径三轮端到端消融为：

| 负载 | baseline | optimized | 变化 |
|---|---:|---:|---:|
| short8 | 50.50 | 50.78 tok/s | +0.55% |
| long16 | 49.69 | 49.57 tok/s | −0.24% |
| short64 | 111.13 | 111.07 tok/s | −0.05% |

这说明“异步 API 存在”没有自动变成“通信被计算掩盖”。当前 `recv_forward` 仍可能阻塞，worker 一次只处理一个消息，没有常驻 receive queue、预投递 receive、独立 stage-level schedule；短请求和在途 microbatch 数不足时，固定调度开销反而会抵消收益。该原型的价值是把优化边界测出来，而不是宣称已经实现 DualPipe。

## Trace 看到的热点

### MoE 明显重于 Attention

在 EP8 短 trace 中，MoE layer range 通常约 **3.2–5.6 ms**，Attention 约 **0.1–0.42 ms**。例如 L37 的 MoE 约 5.61 ms、Attention 约 0.273 ms。MoE range 包含 router、dispatch、NCCL、专家计算和 combine，不能把里面的 NCCL 或子 kernel 再加一次，否则会双计时。

trace 中反复出现三类信号：

1. NCCL SendRecv 和变长 metadata 事件，提示 dispatch/combine 可能受到对端迟到、count 准备和 collective 排队影响。
2. BF16 cublas split-K、小 M/N GEMM，提示每个专家收到的 token 数较小，kernel launch 和矩阵形状效率可能比算力本身更重要。
3. index、sort、scatter、elementwise 等高频小 kernel，以及 CPU 到 GPU 的 count/split 准备，提示 Python 循环和同步点可能制造 host gap。

所以目前最可信的性能假设是“路由和小专家计算的固定开销过大”，而不是直接断言“NVLink 带宽不够”。当前 PyTorch profiler 没有 SM active、Tensor Core active、NVLink bytes 等硬件计数，白色空隙也不能单独证明 GPU 利用率低。

### PP 的气泡和等待

PP2×EP4 中，stage0 负责全局 L00–L23，stage1 负责 L24–L47。stage0 完成某个 microbatch 的 L23 后，stage1 才能开始相应的 L24；如果 stage0 没有立即处理下一个在途 microbatch，stage1 会出现等待。请求数少于流水线深度、单条长请求不能拆成多个 microbatch、prefill 和 decode 混在同一步，都会形成 fill/drain bubble。

现有 trace 只能可靠地说：在完整消息窗口中两个 stage 存在约 20% 左右的普通 kernel 区间并发；同卡计算 kernel 与 NCCL kernel 没有观察到有效交集。它不能支持“zero-bubble”“通信已完全 overlap”或“DualPipe 已实现”。

### 事件缺失本身也是性能分析风险

EP8 的 rank5 kernel 数量约 2.60 万，其他 rank 约 3.25 万；部分消息缺尾层。PP 采样窗口也只覆盖有限 microbatch。缺失事件可能把真实等待误读成 bubble，因此分析器先输出 `common_complete_messages`，再计算 stage 交集；缺失 rank/层的窗口不进入优化排名。这是 trace 质量门槛，不是性能结论。

## 已实现但没有通过生产验收的方案

| 方案 | 为什么做 | 观察 | 当前结论 |
|---|---|---|---|
| `torch_grouped` experts | 减少 Python expert loop 和小 GEMM launch | 早期六类负载约 +3.7%–9.8% | 候选后端；等整模精度、显存、端到端复测 |
| EP ownership | 避免 replicated attention 在每个 rank 重复派发 routes | 理论减少 route 复制，但端到端反而约 −1.5%–7.1% | gather、小 GEMM 和 metadata 抵消收益，默认不宣称有效 |
| source-weight metadata | 在 source 端保留权重，减少一次 metadata collective | 早期 native 路径约 +3.0%–8.2% | 候选优化，需在最终安全数学路径复测 |
| PP dedicated P2P | 让通信有独立 stream 和有限 credit | trace 交集略增，吞吐变化 −0.24% 到 +0.55% | 未形成稳定端到端收益 |
| CPU thread cap | 消除多进程线程过度订阅 | 410–413 s 降到 34.55 s | 已验收的初始化优化 |

EP ownership 的 route 数减少不能直接换算成速度提升：当前实现还要做 gather、恢复 origin 顺序、多个小矩阵计算和 metadata 交换。EP 的变长 all-to-allv 本身允许每个目的 rank 接收不同 token 数，不需要 padding；性能问题在于 count、排序和小 batch 的固定开销。

## 当前安全路径与 vLLM 的差距

在同机、同模型、无重叠的对照中，固定 BF16、eager、prefix caching 关闭、max model len 8192、max batched tokens 512：

| 负载 | vLLM TP8 | vLLM TP4×PP2×EP | 自研 EP8 | 自研 PP2×EP4 | 自研 DP2×EP4 |
|---|---:|---:|---:|---:|---:|
| short128 | 1375 | 1540 | 113 | 105 | 199 tok/s |
| context4k8 | 45.8 | 59.0 | 14.5 | 11.8 | 22.2 tok/s |
| mixed64 | 264 | 332 | 56.9 | 47.3 | 65.1 tok/s |

自研当前安全路径保留参考数学和逐请求 SDPA；vLLM 使用成熟的 TP、paged KV、融合 MoE/attention kernel 和更完整的连续批处理。差距说明下一阶段不能只继续调 NCCL 参数，必须优先补齐 kernel 和调度层能力。上表是当前实现的事实对照，不是声称所有负载的全局最优。

## 下一步如何把热点变成真正优化

### 第一优先级：把 MoE 路由从 Python 控制面搬到设备端

- 用 CUDA/Triton kernel 一次完成 top-k route flatten、expert sort、per-peer counts、local offsets 和 inverse permutation。
- `send_counts.tolist()` 改为设备端 prefix-sum/count buffer，避免 D2H 同步。
- 以 `expert_id` 和 `source_token` 的稳定顺序生成 metadata，保持变长 all-to-allv 的语义，不引入全局 padding。
- 验收指标：CPU launch gap、NCCL start skew、metadata kernel 数、MoE layer p50/p95、route token/s；同时做空 source、极端长尾和不同 top-k 的 oracle。

### 第二优先级：融合 grouped expert GEMM 和 combine

- 先用 CUTLASS grouped GEMM 或 Triton persistent grouped kernel，按本地 expert 的实际 token offsets 直接读取 packed tokens。
- 让 dispatch 输出的 packed buffer 直接作为 GEMM 输入，避免 scatter 到每个专家的临时二维张量。
- 将两个线性层、SiLU/GeGLU 和 weighted combine 尽量合并，减少中间写回。
- 只有在 matched-shape HF logits、EP1/EP8/PP2×EP4 交叉回归通过后，才允许切换默认后端。

### 第三优先级：让 PP 有足够的在途工作

- 将完整 sequence 切成真正的 prefill chunks，并使用 1F1B 或 DualPipe 风格的 warmup/steady/drain schedule。
- 每个 stage 预投递 receive buffer；send/recv 使用独立 P2P stream，主计算 stream 通过 event 等待，而不是 worker 阻塞在 `recv_forward`。
- 用 bounded credit queue 控制激活数量，按 stage 的实测计算时间重新切层，避免单纯按层数均分。
- 验收指标：每个 microbatch 的 stage start/end、首尾 bubble、stage idle、activation bytes、跨 stage NCCL/计算交集，以及无 profiler 的端到端 TTFT/TPOT/吞吐。

### 第四优先级：解决 Attention 和服务调度的结构性差距

- 引入 paged KV 和 FlashAttention/FlashInfer 路径，减少当前复制 attention 输入和逐请求 SDPA 的代价。
- 对在线 mixed workload 做 chunked prefill、长度分桶和 decode 优先级；当前 C256 中 EP8 budget512 的 TTFT P99 约 21.262 s，budget2048 约 6.066 s，说明 token budget/调度策略显著影响尾延迟。
- DP 前端改为独立副本异步忙循环，不让 `step_all` 等待最慢副本；按队列长度、剩余 token 和健康状态路由。

### 第五优先级：用 Nsight 补齐 trace 的因果证据

Perfetto/torch.profiler 适合看层、stage、stream 和 microbatch 关系；要确认“是不是带宽瓶颈、Tensor Core 没吃满、通信是否真正和计算共驻”，还需 Nsight Systems + Nsight Compute：

- Nsight Systems：CUDA API、kernel、NCCL、CPU launch、stream concurrency、NVTX layer/stage 标注。
- Nsight Compute：SM active、eligible warps、Tensor Core utilization、L2/HBM throughput、launch statistics、矩阵 M/N/K。
- NCCL：记录每个 peer 的 payload bytes、send/recv start skew、算法/协议和链路利用率。

## 面试时可以这样回答

“我先不把时间线上的白色空隙直接叫 bubble。先检查 rank、全局层号、microbatch 和共享时钟是否完整，再在完整消息窗口里计算 stage 计算区间交集。trace 显示 MoE 3.2–5.6 ms，明显重于 0.1–0.42 ms 的 Attention，且有大量小 GEMM、索引和 NCCL metadata 事件，所以我优先做 grouped experts 和路由控制面优化；PP 只看到约 20% 的跨 stage 计算交集，同卡通信和计算没有有效交集，于是没有把专用 stream 宣称成 zero-bubble。已经确认的收益是 CPU 线程过度订阅修复，把 DP2×EP4 初始化从约 410 秒降到 34.55 秒。下一步是设备端 prefix-sum/count、融合 grouped GEMM、预投递 receive 和真正的 chunked 1F1B/DualPipe，并用 Nsight 指标和无 profiler 端到端回归验收。”

## 证据文件

- [Profiler 场景分析](Profiler场景分析.md)
- [CUDA Trace 阅读指南](TRACE阅读指南.md)
- [新实验矩阵与尾延迟](新实验矩阵与尾延迟.md)
- [准确性修复与在线服务复核](准确性修复与在线服务复核.md)
- [同机 vLLM 对照与 batch 控制](准确性修复与在线服务复核.md)
- `dp2ep4_default_threads/`、`pp_prefill_baseline/`、`pp_prefill_optimized/`、`rollout_trace_056_20261104_152543/`

最后的边界是：当前项目已经有可解释的性能分析链路和一项明确的初始化优化，但 MoE fused kernel、完整 DualPipe/zero-bubble、全量多模态 serving 和超过 vLLM 的端到端性能仍是后续工作，不应在面试中提前包装成已完成。
