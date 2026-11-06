# vllm-learn：推理并行与性能分析

当前只维护 `v9`。目标模型为 Qwen3-Omni-30B-A3B-Instruct，**自研执行路径目前是文本 Thinker**。这不是完整 Omni 多模态服务，也没有宣称超过 vLLM。

先看 [Profile 操作手册](docs/PROFILE.md)，再看 [真实轨迹分析与优化依据](docs/性能分析.md)。文件太多的问题已整理：实验工具归入 `v9/tools`，自动回归归入 `v9/tests`，可打开的轨迹统一放在 `outputs/profiles`，历史报告不再充当当前使用说明。

| 你想做什么 | 入口 |
|---|---|
| 采八卡 CPU / CUDA / NCCL / 每层轨迹 | `v9/tools/profile_capture.py` |
| 检查漏卡、漏 CUDA、尾层缺失，统计热点与重叠 | `v9/tools/profile_analyze.py` |
| 把八卡叠在同一时间轴，交给 Perfetto | `v9/tools/profile_merge.py` |
| 不开 profiler 测吞吐、TTFT、P95/P99 | `v9/bench.py offline` 或 `serve` |
| 对照 PyTorch / Triton 的 eager 与 CUDA Graph | `v9/tools/kernel_bench.py` |
| 捕获合成反例的 cached 精度差异、逐层定位 | `v9/tools/diagnose_ep.py`、`v9/tools/compare_logits.py`，见 [精度排查](docs/精度排查.md) |

```bash
# 在仓库根目录执行，Linux + CUDA，MODEL 指向已有权重目录。
export MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python v9/tools/profile_capture.py --model "$MODEL" --output outputs/my_pp2ep4 \
  --tp 4 --pp 2 --enable-expert-parallel --requests 32 \
  --prompt-tokens 64 --max-tokens 16 --messages 8 --record-shapes
python v9/tools/profile_analyze.py --input-dir outputs/my_pp2ep4/trace \
  --output outputs/my_pp2ep4/analysis --expected-ranks 8 --layers 48 --require-nccl --same-host
python v9/tools/profile_merge.py --input-dir outputs/my_pp2ep4/trace \
  --same-host --output outputs/my_pp2ep4/trace.json.gz
```

用 [Perfetto](https://ui.perfetto.dev/) 打开 `trace.json.gz`。想先看小文件，请打开 [清理后八卡小模型轨迹](outputs/validation/runtime_cleanup_20261106/final/tiny_pp2ep4/trace.json.gz)；30B 的历史八卡轨迹见 [输出索引](outputs/README.md)。文件齐全与采样完整是两件事，旧轨迹存在尾层缺失，索引中已标明。

其他阅读：[架构与代码顺序](docs/架构.md)、[精度排查过程](docs/精度排查.md)、[环境](ENVIRONMENT.md)、[测试分类](v9/tests/README.md)、[整理与备份](docs/整理说明.md)。

当前 PP 是 V1 风格的 forward-only 多批队列；EP 从 DP×TP 派生；DP 是独立 EngineCore 的推理副本。训练 DualPipe、整模 CUDA Graph、Talker、生产级投机解码不在已完成能力中。历史性能与精度结果按各自源码快照解释。
