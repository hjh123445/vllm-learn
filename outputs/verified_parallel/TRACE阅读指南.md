# CUDA Trace 阅读指南

> 2026-11-05 端到端轨道说明与当前 EP8/vLLM TP8 trace 的采集边界，见[端到端 Trace 阅读指南](端到端Trace阅读指南.md)。目前合并后的 EP8 有 CPU/CUDA/NCCL 轨道；本轮 vLLM TP8 JSON 实际只有 CPU 事件，不能用来比较 GPU overlap。

## 打开文件

推荐用 [Perfetto UI](https://ui.perfetto.dev/) 或 Chrome tracing 打开整理后的 `*_gpu_timeline.json`。原始 PyTorch profiler JSON 更大但没有我们的中文 track metadata。SVG 是快速总览；精确 event、stream、层号和 microbatch 请在 JSON 时间线里点事件看 args。

| 文件 | 内容 |
|---|---|
| `../qwen3_omni_trace.json` | 较早的 1 GPU、48 层、decode 文本路径；没有 PP stage |
| `ep8_trace_gpu_timeline.json` | EP8、多 GPU、短 decode profile，逐 stream/逐 GPU |
| `pp2ep4_trace_gpu_timeline.json` | PP2×EP4、多 GPU、短 decode profile |
| `pp_prefill_baseline/trace/cuda_rankN.json` | 196–197token prefill 的部分层采样；文件rank0–7齐全，但rank6没有kernel |
| `pp_prefill_optimized/trace/cuda_rankN.json` | 同 workload 的异步P2P部分层采样；8rank有kernel，不是完整prefill |
| `pp_prefill_baseline/trace/pipeline_rankN.json`、`pp_prefill_optimized/trace/pipeline_rankN.json` | 主机侧 submit/transport trace；不是 CUPTI GPU 时钟 |
| `pp_prefill_baseline_full/trace/` | 不完整补充采集：缺少 rank2、rank0 有 `.partial` 文件且 host trace 不齐，不用于跨 rank 对照 |
| `../pp_cuda_event_baseline.json`、`../pp_cuda_event_optimized.json` | 两卡 synthetic stage timeline；`attention_like`/`moe_like` 是合成 CUDA event，不是 Qwen 模型 kernel；`host_p2p` 是请求/API 生命周期 |
| `../pp_transport_benchmark.json` | 两卡 synthetic transport microbenchmark；Profiler 没有 CUDA kernel-category event，不能据此量化真实 GPU overlap |
| `*_gpu_timeline.svg` | 对每个 GPU/CUDA stream 分行的 kernel overview |

浏览器网页版 Perfetto 若不允许上传内部文件，可用本地 Perfetto Desktop/UI；这些 trace 中只有合成文本负载和模型算子活动，没有输入 prompt 字符串。

## Track 与标注

- process：`GPU N / rank N / PP stage S / EP rank E`。EP8 全是 `stage0`；PP2×EP4 中 rank 0–3 是 stage0，rank 4–7 是 stage1。
- thread：CUDA `stream ID`。通信 kernel 和计算 kernel 可能在不同 stream；不要把线程号误认为 CPU thread。
- Attention：青绿色 `L00 注意力` 对应模型全局 layer 0 的 self-attention GPU annotation/kernel launch。
- MoE：红色 `L00 MoE` 对应同一 global layer 的 router/dispatch/expert/combine 区域。
- kernel tooltip 保留原 CUDA kernel 符号名；args 有 `layer_global`、`op_type`、`microbatch_id`、`lane`、`phase`、`device`、`rank`、`pp_stage`、`ep_rank`。kernel 的层归属通过 CUPTI External id 回连 CPU `record_function` launch range；无法匹配时保留原名和未归属状态。
- 蓝色为 NCCL kernel；黄色为 memcpy/memset。NCCL 是实际 GPU kernel 活动，但 duration 可能含轮询/等待，不等于纯链路搬运时间。

## 两个主要多卡场景

```text
EP8：每一层仍在 GPU0..7 上；每卡有 16/128 个专家。
GPU0..7: Attention(L00) -> EP dispatch -> 本地 MoE(L00) -> EP combine -> ... L47

PP2 x EP4：层按全局层连续切开。
GPU0..3 / stage0: embedding -> L00..L23 Attention/MoE -> activation P2P
GPU4..7 / stage1: activation P2P -> L24..L47 Attention/MoE -> norm/logits/sample
```

EP trace 适合看每层 All-to-All 和本地专家计算的设备差异。PP trace 适合看 stage 之间 activation P2P 是否和相邻 stage 的计算真正同时发生。PP2×EP4 不能直接拿 event 数或 annotation sum 与 EP8 相除，因为 layer partition、profile 窗口和通信路径都不同。

## 旧单卡与当前多卡

旧 `qwen3_omni_trace.json` 里一个 worker process/一个 GPU/一个 stream，层标注 48 层各 3 次。它只能回答单卡 decode 样本中 attention/MoE 的 kernel 构成，回答不了多卡 EP 通信、PP 空泡或异步 overlap。

当前多卡 trace 含 profiler 的 kernel、NCCL 和 stream id。EP8 短decode的rank5缺少GPU kernel，不能用它判断rank5计算量。prefill baseline/optimized分别位于 `pp_prefill_baseline/` 和 `pp_prefill_optimized/`，文件rank0–7齐全，但baseline的rank6没有kernel；两者stage0约采到L0–18、stage1约L24–40，属于部分层窗口。共享 `baseTimeNanoseconds` 允许保留跨rank原始时钟，却不能弥补采样缺失。`pp_prefill_baseline_full/` 也不完整；不要拿缺失事件的空白当GPU bubble。

PP2×EP4 stage 1 的 CUDA trace起点比 stage0晚约 137ms。各 rank 都有一致的 `baseTimeNanoseconds=1790857026000000000`，所以应保留原始 ts；若各文件独自归零，会造出不存在的 overlap。

注意区分两组“pipeline trace”：`pp2ep4_trace_gpu_timeline.json` 与 `pp_prefill_*/trace/cuda_rank*.json` 来自 Qwen3-Omni 实际模型 forward；`../pp_cuda_event_*.json` 是合成调度示意，其中的 attention/moe 时长由 synthetic workload 生成。后者不能用于报告 Qwen 每层耗时，也没有真实层号映射。

## 各类事件的正确解释

1. `user_annotation` 是 CPU 线程发出的范围；`gpu_user_annotation` 是对应 GPU 时间范围。`kernel` 才是 CUDA kernel 区间。不同类别的范围常互相嵌套。
2. 同一 layer 的 MoE 总 range 包括 router、dispatch、expert、combine 等子区间；不要把所有 annotation 时长相加。NCCL range/Kernel 嵌在 MoE 中也不能额外加一次。
3. 查看 GPU span 内没有 kernel 的白色空隙，是当前采样中的 GPU 无 kernel interval；它可能源自主机 launch 间隔、GPU 同步、NCCL wait、stage 依赖、profile 窗口边缘，不可仅凭空隙断言是哪一类。对照 CPU annotation、comm stream 和相邻 rank。
4. `compute_nccl_overlap_ms` 是 kernel 区间集合交集；本窗口如果为 0，只能说采样里未观测到计算 kernel 与 NCCL kernel 的时间相交。kernel overlap 不代表有效 SM overlap；反过来，区间有交集也需要 Nsight/硬件指标证明实际资源共驻。
5. `kernel_union_fraction` 是所有 kernel interval union / first-to-last GPU event span，便于扫描 gap；不是 GPU utilization、occupancy 或忙碌 SM 百分比。
6. 主机 `pipeline_rankN.json` 的 clock 是 shared host monotonic `perf_counter`；CUDA profiler trace 是 CUPTI 的时间基准。不能拼在同一时间轴上。先单独看 GPU stream，再用 host trace理解 recv阻塞和 send 提交顺序。

## 对比 baseline 和 optimized

打开 `pp_prefill_baseline/trace/cuda_rank*.json` 与 `pp_prefill_optimized/trace/cuda_rank*.json`，看相同 rank、相同 stream 和相同 `microbatch_id`。重点记录：

- stage0 每个 microbatch 的最后一个层 kernel何时结束、P2P/NCCL何时开始；
- stage1相应 activation到达后何时开始 L24，并在其 compute 期间 stage0 是否已开始下一 MB；
- NCCL kernel有没有跨到计算 kernel之上，以及 P2P等待是否仍把接收 stage空转；
- 每个stage时间窗内的 kernel union、NCCL overlap、首尾 bubble 和每条消息尺寸；
- baseline与optimized的端到端任务 tok/s/TTFT/P95 是否变化。

optimized 多一个通信 stream，不代表重叠成功。同口径三轮结果中，optimized 相对 baseline 的 short8/long16/short64 吞吐变化约+0.55%/−0.24%/−0.05%，原始数据在 `pp2ep4_threads4/results.json`。该证据不支持稳定收益，更不能称零气泡。

独立 synthetic transport benchmark 也没有显示异步方案收益：两张 A800 上，两个 activation shape 的 optimized median 分别慢38.5%和10.1%。它只能暴露 prototype transport 的固定开销风险；该 benchmark 没有采到 CUDA kernel-category 事件，不能当作真实模型的 GPU overlap 测量。

## 限制

PyTorch profiler/CUPTI trace 没有 SM active、Tensor Core active、内存带宽占用、NVLink 链路 bytes、PCIe replay 等完整硬件指标；看详细瓶颈应补 Nsight Systems + Nsight Compute。短 trace是 profiler 抽样，不是全生成生命周期。microbatch数、请求到达、设备进程调度也影响流水线空泡。
