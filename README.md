# vllm-learn：Qwen3-Omni-30B 推理引擎实验

仓库现在只保留当前可运行的 v9 实现，目标是研究 Qwen3-Omni-30B-A3B-Instruct 在 A800 上的调度、TP、EP、PP、DCP、DP、量化和性能分析。早期 v0–v8 的独立实现已从工作树删除，完整历史仍可通过 Git 提交查看。

## 运行入口

```bash
cd v9
python example.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct
python server.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct --tp 1 --ep 8
```

核心执行路径是：`EngineCore → Scheduler → WorkerProc → ModelRunner`。多卡启动时每张 GPU 一个 worker；DP 使用 `DPFrontend` 启动多个相互独立的 EngineCore。并行组件位于 `v9/layers/parallel.py`、`v9/layers/moe.py`，模型文件不需要被额外的 parallel.py 劫持。

## 目录

```text
v9/
  engine/       EngineCore、scheduler、worker、PP activation transport、DP frontend
  layers/       TP/EP 组件、paged attention、DCP、量化和 CUDA/Triton kernel
  models/       Qwen3-MoE/Qwen3-Omni thinker 模型
  tests/        当前 v9 的单元测试、矩阵实验、通信基准和 trace 工具
  utils/        配置和 rank-local safetensors loader
```

## 当前实现边界

- EP 使用真实 `send_counts/recv_counts` 的变长 all-to-allv，不做固定容量 padding。
- PP2/PP4 和 PP2×TP2 已通过端到端验证；当前 transport 仍是阻塞式 P2P 正确性基线，DualPipe/Zero-Bubble 尚未完成。
- DP 是推理副本并行，每个副本拥有独立 EngineCore、KV cache 和通信组；它不等同于训练 DDP。
- 投机解码原型已移除出运行路径，当前不宣称已实现生产级 speculative decoding。
- W4 量化保留为实验路径，当前速度和质量仍不能视为生产后端。

## 实验报告与 profile

中文报告、实验摘要和选定原始记录现位于 [outputs/verified_parallel](outputs/verified_parallel/交付说明.md)。
先读[当前结论与证据边界](outputs/verified_parallel/README_面试实验报告.md)和[精度问题排查过程](outputs/verified_parallel/精度问题排查方法论.md)。完整本地/远程归档还包含未纳入 Git 的大规模逐层张量和逐 rank 原始 trace，公开文件清单见交付说明。

- 多卡采集入口：[capture_parallel_window.py](v9/tests/capture_parallel_window.py)。
- CPU/CUDA profiler 实际在每个 worker 中启用：[worker.py](v9/engine/worker.py)。
- [EP8 八卡 CPU/CUDA/NCCL trace](outputs/verified_parallel/trace_comparison_056_20261105/custom_ep8_compact_cpu_cuda.json.gz)；[rank0 轻量版](outputs/verified_parallel/trace_comparison_056_20261105/custom_ep8_rank0_cpu_cuda_preview.json.gz)。下载后可直接尝试 Perfetto；不识别 gzip 时先解压成 JSON。
- 采集命令与环境：[queue.json](outputs/verified_parallel/rollout_trace_056_20261104_152543/queue.json)；分析脚本见 [tools/experiment_analysis](tools/experiment_analysis)。

当前默认参考数学路径在匹配执行条件的 12×32 个固定前缀位置上对齐 HF；默认批处理仍存在跨形状 BF16 数值敏感性，融合/Triton 路径尚未通过完整模型质量验收。不能把候选吞吐数字视为已通过全部质量门槛的生产性能。
