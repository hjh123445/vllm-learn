# 端到端 Trace 怎么看：CPU、CUDA、NCCL 与 Perfetto

## 先回答：时间线里应该有哪些“条”

常见的 GPU 端到端时间线不是只有一条 CPU 时间和一条 CUDA 时间。它通常按**进程、CPU 线程、GPU 设备、CUDA stream**分轨显示：

| 轨道 | 里面是什么 | 能回答什么 |
|---|---|---|
| 服务端 CPU 线程 | HTTP/RPC、tokenizer、请求排队、scheduler、Python/C++ 算子范围、锁等待、CUDA API 调用 | 请求在哪个线程排队/阻塞，CPU 是否来不及喂 GPU，哪个 API 发起了 GPU 工作 |
| GPU N 的 CUDA stream S | 真正开始执行的 CUDA kernel、GPU memcpy/memset；NCCL GPU kernel也在某个 stream 上 | GPU 实际何时执行、不同 stream 是否有时间交集、GPU 工作之间的空档 |
| 用户标注 / NVTX / `record_function` | request、prefill/decode、层号、attention/MoE、microbatch 等逻辑范围 | 把底层 kernel 对回模型层、请求阶段和调度单元 |
| 多卡 rank / PP stage | 每个 GPU worker 的 CPU/GPU 轨道，以及 stage/microbatch 标签 | rank 是否失衡、stage 是否空等、activation 传输前后有没有空泡 |

**CUDA API 调用在 CPU 轨道上；kernel 才在 GPU stream 轨道上。** CPU 发出 `cudaLaunchKernel` 后通常马上返回，GPU 工作仍在队列里运行；所以 CPU 上的 `aten::mm` 或 launch API 的长度，不等于 GPU 上矩阵乘实际运行时间。CUDA event 的 correlation / external id 用来把 launch 和 kernel 对起来。

`stream 7` 是设备内的 CUDA 执行队列编号，不是 CPU 的第 7 个线程，也不代表一张 GPU 只有一个 SM。不同 stream 的 kernel 时间重叠，才说明调度上有并发机会；最终是否实际共驻、是否争用 SM/带宽，仍需 Nsight Compute 或硬件计数器确认。

真正的“端到端”还应在模型轨道之外加入请求生命周期：`客户端发送 → 服务端接收 → 排队 → tokenize/media preprocess → batch/scheduler → prefill → decode/token stream → detokenize/返回客户端`。Perfetto 只负责展示采到的事件；如果服务端没有写这些 span，打开 GPU kernel trace 并不会自动得到 HTTP 端到端延迟。

## 网上常见的采集与阅读方式

PyTorch Profiler 的基本做法是同时启用 `ProfilerActivity.CPU` 和 `ProfilerActivity.CUDA`，用短的、明确的 profile window 导出 Chrome trace，再在 Perfetto 看 CPU op、CUDA runtime 和 kernel 轨道。CUDA 事件由 CUPTI 采集；只有 CPU activity 时，导出的 JSON 不会凭空出现 GPU stream 时间线。

Nsight Systems 的时间线把 CPU threads、CUDA API 和 GPU device/context/stream 活动放在一张图里，并能把 CPU launch range 与 GPU 执行活动对应。vLLM 官方 profiling 文档建议性能敏感的低开销分析优先用 Nsight Systems；PyTorch Profiler 适合带更多形状、调用栈等调试信息的中等开销分析。两者都不应直接用于正式吞吐数字，因为采集会改变调度和耗时。

Perfetto 是查看与分析 trace 的 UI/trace processor，不是采集器。采集端必须先把 CPU、CUDA、NCCL、NVTX/模型标注正确写进 trace，再用 Perfetto 打开。

## 在 Perfetto 里照着看

1. 打开 [Perfetto UI](https://ui.perfetto.dev/)，选择 **Open trace file**，载入本指南下方的 EP8 `.json`。建议用本地 Perfetto Desktop；文件约 127 MiB，浏览器内存不足时会长时间加载。
2. 左侧先找到 `GPU 0 / EP rank 0 / custom EP8` 到 `GPU 7 / EP rank 7 / custom EP8`。每个 GPU 进程下有 `CPU API / annotations` 和 `CUDA stream 7 | GPU N | kernels + NCCL` 两条主要轨道。
3. 按 `F`/选择一段时间放大到一个 `stage0.layerN.attention` 或 `stage0.layerN.moe`。CPU 条看 Python/C++ 范围和 runtime launch；CUDA 条看真实 kernel 起止。点击事件，在右侧 details 查看 `rank`、`gpu`、`stream`、`layer_global`、`op_type`、`correlation`、`Collective name` 等字段。
4. 先看一个 rank，再看八张卡同一个层/collective 的对齐情况。最慢 rank 的结束时间通常约束 collective 或下一阶段；不能只看 rank0。
5. 查 overlap 时，在同一 GPU 的 CUDA stream 轨道上比较普通计算 kernel 与 NCCL kernel 的时间区间；跨 GPU 同时出现 collective 不等于该 GPU 上“通信与计算重叠”。再回 CPU 轨道检查 launch 间隔、`cudaStreamSynchronize`/事件等待和上层 MoE 范围。
6. 将 trace 结论与**未开 profiler**的客户端 TTFT、TPOT、吞吐、P95/P99 并列。trace 用来定位机制，benchmark 用来证明用户可见的收益。

### 如何严谨判断通信和计算重叠

对同一设备上计算 kernel 的区间集合 `C` 与 NCCL kernel 的区间集合 `N` 求交集：

```text
overlap_us = Σ（compute interval 与 NCCL interval 的时间交集）
```

若 overlap 为 0，只能说这段采样里没有观测到同设备的 compute/NCCL kernel 时间相交；不能推成所有负载都不重叠。若大于 0，说明时间线上有区间交叠，不足以证明有效并发或链路带宽被充分利用。还需检查 stream、依赖事件和 GPU 资源指标。NCCL GPU kernel 的 duration 可能包含协议轮询/等待，不等于线缆实际搬运数据的纯耗时。

## 本仓库现有两个对照文件各是什么

### 自研 EP8：CPU + CUDA + NCCL 都在

- 可直接导入 Perfetto：[`custom_ep8_compact_cpu_cuda.json`](trace_comparison_056_20261105/custom_ep8_compact_cpu_cuda.json)
- 首次试开的小文件：[`custom_ep8_rank0_cpu_cuda_preview.json`](trace_comparison_056_20261105/custom_ep8_rank0_cpu_cuda_preview.json)（单 GPU、约 16 MiB，先确认 Perfetto 能显示 CPU 与 CUDA 两条轨道，再开八卡版）
- 事件统计：[`custom_ep8_compact_cpu_cuda.summary.json`](trace_comparison_056_20261105/custom_ep8_compact_cpu_cuda.summary.json)
- 原始八个 rank trace：[`rollout_trace_056_20261104_152543/trace_ep8/trace`](rollout_trace_056_20261104_152543/trace_ep8/trace)
- 这是 `short64` profile run 中每个 rank 的首个 `stage0.message1.mb0.prefill` 窗口，约 535 ms；八条请求、输出仅 4 token 的 trace profile，**不是服务端 HTTP 全生命周期，也不是吞吐测量**。profile 会改变调度，不能从这条轨迹直接报 tokens/s。
- 合并文件包含八个 rank、CPU annotations/API、CUDA kernel、memcpy/memset 和 NCCL kernel；共 249,549 个事件，约 127 MiB。八张卡在这个窗口中都只记录到 `stream 7`。每张卡约 10.0k 个普通 kernel 和 240 个 NCCL kernel。
- 对同一 GPU 的普通 kernel 与 NCCL kernel做区间求交，八个 rank 均为 0；并且通信、计算在同一个 stream 上，按 stream 顺序执行。**这个样本不支持“EP 的通信已和计算重叠”**。它提示后续要有独立通信 stream/异步依赖，还要证明收益；光创建第二条 stream 也不代表会重叠。
- 轨道 metadata 已修正：CPU 与 CUDA track 使用不同 tid，metadata 与事件 tid 一致，避免 rank 7 CPU tid 和 stream tid 冲突造成轨道重名/错名。

### vLLM TP8：现有 JSON 只有 CPU 事件，不能做 GPU overlap 对照

- 目录：[`vllm_tp8/raw2`](trace_comparison_056_20261105/vllm_tp8/raw2)
- rank0 JSON 有 51,946 个 `cpu_op` 和 18 个 CPU annotation，但没有 `kernel`、GPU memcpy 或 NCCL GPU kernel 类事件；其余 rank 文件同样是 CPU-side trace。`profiler_out_0.txt` 是 profiler 汇总表，不是 GPU 时间线。
- 尽管采集配置选择了 vLLM torch profiler，产物实际不含 CUDA 活动。因此现在可以用该文件看 CPU 算子/调用结构，**不能**拿它比较 vLLM TP8 与自研 EP8 的 GPU kernel 时长、CUDA stream 或通信重叠，也不能把 `vllm::all_reduce` 的 CPU 范围当作 NCCL GPU kernel。
- vLLM 官方当前文档写明 NVIDIA 上 torch profiler 默认活动为 CPU+CUDA，但现场 JSON 与该预期不符。准确原因尚未从产物定位；重采时应显式设置 `torch_profiler_activities=["CPU","CUDA"]`，并在导出后先校验确有 `kernel`/device/stream 事件；仍缺失则用 Nsight Systems。已有文件不能事后补出 GPU 时间线。

## Perfetto 打开空白时的排查顺序

1. 确认打开的是实际 `.json` trace，不是 `.summary.json`、SVG、空目录或尚未完成写入的 `.partial`。
2. 检查 JSON 顶层有 `traceEvents` 数组、事件有 `ph/ts/dur/pid/tid`，文件不是 0 byte，能被 JSON parser 完整解析。
3. 先用一个很小的 trace 检验 Perfetto 版本/UI，再加载大文件；此 EP8 合并文件约 127 MiB，浏览器可能在解析时占用数倍内存。
4. 检查 `process_name`/`thread_name` 的 `(pid,tid)` 是否与事件一致且无重复；检查时间戳是否保留各 rank 的共同时间基准，不能每个 rank 单独从 0 开始，否则会制造假的 overlap。
5. 看左侧是否有 CPU thread 和 GPU stream track。若只有 CPU track，不是 Perfetto 把 GPU 藏起来了，而是采集文件里没有 GPU 活动；确认 PyTorch profiler 活动列表含 CUDA，或改用 Nsight Systems。
6. 旧 trace 缺 rank、CUDA kernel、GPU stream 事件，或不同 rank 使用不兼容的时钟时，不要把空白当成 GPU bubble，也不要拿来计算跨 rank overlap。

## 下一轮“真正端到端”采集应补什么

针对动态在线负载，给每个请求写匿名 `request_id` 和时间标记：服务接收、队列进入/离开、tokenize/media preprocess、scheduler batch、prefill start/end、每个 decode step、首 token flush、每次客户端 flush、最后 token。将 `request_id → batch_id → microbatch_id → PP stage → layer → CUDA correlation` 通过 NVTX/record_function 与 CUPTI/Nsight 的时间线连起来。

每个 workload 都记录到达模式、并发数、输入/输出 token 长度分布、batch/token budget、模型版本、并行配置、GPU 型号、软件版本和 trace 区间。至少分别跑短输入/长输入/长短混合及并发度梯度。正式吞吐与延迟用无 profiler 客户端计时；profile run 单独跑、短采样、保留完整环境信息。这样才可以回答“模型计算是瓶颈还是服务排队、前处理、调度、通信、GPU launch 空档造成的端到端变慢”。

## 参考资料

- [NVIDIA Nsight Systems User Guide — Timeline View / NVTX ranges](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：CPU thread、CUDA API、GPU stream 和 NVTX 标注的时间线阅读方式。
- [PyTorch Profiler 文档](https://docs.pytorch.org/docs/stable/profiler.html) 与 [Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：如何选择 CPU/CUDA activities、导出 trace，并理解 profiler overhead。
- [Perfetto UI 文档](https://perfetto.dev/docs/visualization/perfetto-ui)：导入、浏览、处理大型 trace 的方法。
- [vLLM Profiling 文档](https://docs.vllm.ai/en/latest/contributing/profiling/)：vLLM 的 torch profiler、Nsight Systems 用法；文档建议只 profile 少量请求，并提醒 profiling 会显著拖慢推理。
