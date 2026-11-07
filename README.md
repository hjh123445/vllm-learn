# vllm-learn：推理并行与性能分析

当前只维护 `v9`。目标模型为 Qwen3-Omni-30B-A3B-Instruct，自研执行路径支持文本 Thinker，以及可选的静态图片理解和文本输出。音频、视频、Talker 和 Code2Wav 尚未接入。

2026-11-08 已完成 TP8+EP 下 128/256 并发、在线/离线、文本/图像三档负载，共 26 档、78 轮、29,952 请求，零错误。正常 EOS 的 ChartQA 256 题 Graph/eager 完整输出 256/256 相同；HF 对照为 250/256 相同。详见 [最终验证与负载总表](docs/CUDA_Graph多模态最终验证报告.md)、[CUDA Graph 与多模态原理](docs/CUDA_Graph与多模态原理.md)。

先看 [Profile 操作手册](docs/PROFILE.md)，再看 [真实轨迹分析与优化依据](docs/性能分析.md)。文件太多的问题已整理：实验工具归入 `v9/tools`，自动回归归入 `v9/tests`，可打开的轨迹统一放在 `outputs/profiles`，历史报告不再充当当前使用说明。

| 你想做什么 | 入口 |
|---|---|
| 普通服务/离线运行顺便采各卡 CPU / CUDA / NCCL / 每层轨迹 | `v9/server.py --profile` 或 `v9/bench.py offline --profile` |
| 检查漏卡、漏 CUDA、尾层缺失，统计热点与重叠 | `v9/tools/profile_analyze.py` |
| 把八卡叠在同一时间轴，交给 Perfetto | `v9/tools/profile_merge.py` |
| 不开 profiler 测吞吐、TTFT、P95/P99 | `v9/bench.py offline` 或 `serve` |
| 对照 PyTorch / Triton 的 eager 与 CUDA Graph | `v9/tools/kernel_bench.py` |
| 捕获合成反例的 cached 精度差异、逐层定位 | `v9/tools/diagnose_ep.py`、`v9/tools/compare_logits.py`，见 [精度排查](docs/精度排查.md) |

```bash
# 在仓库根目录执行，Linux + CUDA，MODEL 指向已有权重目录。
export MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python v9/bench.py offline --model "$MODEL" --output outputs/my_pp2ep4 \
  --tp 4 --pp 2 --enable-expert-parallel --max-batch-size 128 \
  --workloads short128 --repeats 1 --profile
python v9/tools/profile_analyze.py --input-dir outputs/my_pp2ep4/trace \
  --output outputs/my_pp2ep4/analysis --expected-ranks 8 --layers 48 --require-nccl --same-host
python v9/tools/profile_merge.py --input-dir outputs/my_pp2ep4/trace \
  --same-host --output outputs/my_pp2ep4/trace.json.gz
```

用 [Perfetto](https://ui.perfetto.dev/) 打开 `trace.json.gz`。最新样例是 [八卡全程小模型轨迹](outputs/validation/profile_fullrun_20261106/tiny_pp2ep4/trace.json.gz)，各卡16条消息逐条验收通过；30B 的历史八卡轨迹见 [输出索引](outputs/README.md)。文件齐全与采样完整是两件事，旧轨迹存在尾层缺失，索引中已标明。

其他阅读：[架构与代码顺序](docs/架构.md)、[精度排查过程](docs/精度排查.md)、[环境](ENVIRONMENT.md)、[测试分类](v9/tests/README.md)、[整理与备份](docs/整理说明.md)。

当前 PP 是 V1 风格的 forward-only 多批队列；EP 从 DP×TP 派生；DP 是独立 EngineCore 的推理副本。已完成的是 TP/EP、PP1、dense-EP 的 decode-only CUDA Graph（默认严格 batch-invariant 数值策略）；PP/DP 整模 Graph、训练 DualPipe、Talker、生产级投机解码不在已完成能力中。历史性能与精度结果按各自源码快照解释。
