# DP2×TP2 / EP4：cached generation 单 token 分歧

详细方法、逐层定位、数学原理和复现命令见 [中文精度报告](../../../docs/精度排查.md)。
本目录只有合成 8 层小模型的正确性证据，不是 Qwen3-Omni-30B 性能或质量成绩。
四组主对照使用 A800 GPU 0–3、Torch 2.9.1+cu130、CUDA 13.0、驱动 580.65.06。
完整依赖版本及实验结束后的任务进程核对见
[环境审计](provenance/dependency_versions.json)；`task_processes_remaining=[]`。

## 先看什么

| 文件 | 内容与结论 |
|---|---|
| [summary.json](summary.json) | 主结果、测试统计和未完成边界 |
| [bf16_comparison.json](bf16_comparison.json) | 默认路径：DP1/call003/seq0，token57 与 token54 变为并列，argmax 改变；保留失败 |
| [precise_comparison.json](precise_comparison.json) | 仅专家 FP32：20 个正式生成 token 相同，24 行已捕获 logits 逐位相同 |
| [replay.json](replay.json) | 32 个层批次的逐 route dispatch/专家/返回合并精确重放，以及第 0 层四种精度消融 |
| [replay.py](replay.py) | 上一行的可执行复核脚本；只适用于本反例的 E4/EP4/TP2/H16/I16 |
| [fp32_comparison.json](fp32_comparison.json) | 较早全模型 FP32 辅助对照；也改变了权重加载精度，不作为单变量因果主证据 |

比较报告的 `records` 是相同前缀的完整词表 logits 指标、真实 sampler 输出、margin/tie；
`module_boundaries` 是整批匹配时各层 attention/router/MoE 输出；`skipped` 是前缀已经
分叉而不能比较的行。`accepted_generation=false` 表示未完成 chunk-prefill 的临时采样，
不计为正式生成位置。EP 多出的空闲 DP 调用没有 sampler，单独记入 `idle_collective_calls`。

## 原始数据与来源

`raw/` 保留压缩包，避免数百个张量文件散落仓库：

| 包 | 实际执行 |
|---|---|
| `final_bf16_tp.tar.gz` | DP2×TP2，关闭 EP，默认模型精度 |
| `final_bf16_ep.tar.gz` | 相同 DP/TP，开启 EP4，默认模型精度 |
| `final_precise_tp.tar.gz` | 关闭 EP，仅专家算术/加权/归约 FP32，模型及权重存储仍 BF16 |
| `final_precise_ep.tar.gz` | 开启 EP4，仅专家 FP32 |
| `fp32_tp.tar.gz`、`fp32_ep.tar.gz` | 较早全模型 FP32 辅助实验；当时尚无逐次 source manifest |
| `capture_source.tar.gz` | 四组主对照共同且经哈希验证的完整 v9 Python 采集源码，可在独立目录解压 |
| `synthetic_checkpoint.tar.gz` | 实际生成并加载的小权重、config 与 tokenizer |

每个主实验包中：`manifest.json` 保存启动前的源码/权重 SHA256 和参数；
`results.json` 保存实际输入输出与环境；`dpX_tpY/callNNN.pt` 保存该进程每次运行的
批次、KV 元数据、模块边界和 sampler 张量；`weights_layerN.pt` 保存实际本地专家切片。
PyTorch 张量用 `torch.load(..., map_location='cpu', weights_only=True)` 读取。
诊断会同步 GPU，不能用日志耗时报告吞吐。

四个主实验 manifest 的源码和权重哈希完全一致，由
[seal.py](provenance/seal.py) 复核并封存。下载后对 34 个远端文件逐一验证 SHA256。
远端原始哈希清单在 `provenance/remote_sha256.json`，其 key 使用远端打包前的平铺文件名；
本地整理后的清单为 `sha256.json`。`provenance/final_repository_source_sha256.json`
对应最终仓库 Python 文件；采集后分析器只增加了一个汇总字段，采集源快照单独保留。

## 命令与回归

* [verified_queue.sh](provenance/verified_queue.sh)：四组主对照的实际队列，含依赖路径、线程数和超时。
* [final_checks.sh](provenance/final_checks.sh)：默认策略的三个真实生产拓扑回归及相关单元回归。
* [precision_regression_command.txt](provenance/precision_regression_command.txt)：7 项精度回归的原执行命令，保留其退出码记录缺口。
* `logs/default_regression.log/.exit`：2 passed、1 failed，pytest 退出码 1；原严格失败未改。
* `logs/precision_regression.log`：新 FP32 专家生产生成/单侧 idle-DP 回归和六项组件回归，7 passed。
  最初命令通过 `tail` 返回，未单独保存原 pytest 退出码，不能把 wrapper 的 0 当作 pytest 0；
  这里依据 pytest 日志的完整 `7 passed` 终止摘要报告。
* `logs/unit_regression.log/.exit`：20 passed、12 subtests passed，退出码 0。
* `logs/analysis_regression.log/.exit`：2 passed，退出码 0。
* `logs/replay.log/.exit`、两份主 comparison 的 `.log/.exit`：均正常结束、退出码 0。

前两次诊断工具开发失败也保留在 `logs/bf16_tp*`：第一次 runner 构造签名未接收 dtype，
第二次误以为 worker 的 `_Spec` 含完整 token_ids；实际只含当前 `_tokens` window。
修正后由诊断器自行重建历史，没有向生产协议增加字段。这两次不是模型精度结果。
分析器首次也误把 EP-only 空闲调用当作缺失 capture，现已分开计数并加入离线回归；
`final_checks.sh` 中分析步骤当时失败，最终有效结果以各 comparison 的独立退出码为准。

解压四个主实验包到同一个目录，把 `replay.py` 放到该目录后：

```bash
python /path/to/v9/tools/compare_logits.py --cached-reference final_bf16_tp \
  --cached-actual final_bf16_ep --output bf16_comparison.json
python /path/to/v9/tools/compare_logits.py --cached-reference final_precise_tp \
  --cached-actual final_precise_ep --output precise_comparison.json
# GPU 重放 BF16 GEMM，不能替换成 CPU 结果。
python replay.py --reference final_bf16_tp --actual final_bf16_ep
```

默认 BF16 跨拓扑严格生成一致性仍未通过。高精度策略只证明这个反例得到消除；
没有测真实 30B 的性能代价或任务质量，也没有将全模型 FP32 的辅助结果冒充单变量修复。
