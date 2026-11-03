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

实验数据和中文分析报告位于仓库外的 `outputs/` 交付目录。
