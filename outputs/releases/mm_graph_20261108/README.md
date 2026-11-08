# 最终交付索引

本目录随 GitHub 版本发布。先看 [版本/服务器副本审计](../../../docs/最终版本与服务器副本核对.md) 和 [复现说明](../../../experiments/mm_graph_20261108/README.md)。服务器可访问范围共发现 68 份代码副本，保留原目录；当前主线没有合并带已知数值限制的旧 fastmeta 双通信器实验。发布前默认回归 169 passed / 10 skipped；显式 NCCL 生产回归 5 passed / 1 failed，失败为已知 DP2×TP2 默认 BF16 严格 greedy 反例，见 [原始日志](release_audit/nccl.log)。不能将本发布理解为所有并行拓扑都已通过。

2026-11-08。实验：单机 8×A800、Qwen3-Omni-30B-A3B-Instruct、BF16、TP8+EP。

先阅读 [最终验证与负载总报告](cuda_graph_multimodal_final_report.md)，再按需阅读 [CUDA Graph 与多模态原理、修复详情](cuda_graph_and_multimodal_explainer.md)。后者解释 batch-invariant、每 GPU rank/每 bucket 的捕获、图像和 Thinker 的关系、HF/vLLM 接入差异。

最终结果：26 档 × 3 轮 = 78 轮、29,952 请求，零错误；25,344 次跨条件 token 比较零差异；正常 EOS 的 256 题 Graph/eager 输出 256/256 相同。HF 对照仍为 250/256 相同，当前准确率 216/256、HF 217/256，六题差异逐项保留。真实视觉特征 768 组均逐位一致。严格 logits/KV 回归同时覆盖短上下文和到 2,048 的容量边界。

| 文件 | 用途 |
|---|---|
| [load_matrix_detailed.csv](load_matrix_detailed.csv) | 52 行首轮/后续轮数据，p50/p95/p99、吞吐、实际 prompt 长度和显存 |
| [load_matrix_all_rounds.json](load_matrix_all_rounds.json) | 全部 78 轮聚合指标，保留首轮 |
| [graph_execution_summary.json](graph_execution_summary.json) | 各进程组的 rank 捕获/回放数及 Graph pool |
| [v9_final_verified_source.zip](v9_final_verified_source.zip) | 最终验证源码和复现实验脚本 |
| [final_benchmark_evidence.tar.gz](final_benchmark_evidence.tar.gz) | 原始证据归档，已校验 SHA256；下载后自行解压 |
| [artifact_manifest.json](artifact_manifest.json) | 两个最终归档的大小和 SHA256 |
| [final_source_manifest.json](final_source_manifest.json) | 实验服务器/源码 zip 的原始逐文件 SHA256 |
| [git_source_manifest.json](git_source_manifest.json) | Git blob 指纹；87 文件与实验源码仅 CRLF/LF 不同，107/107 规范化内容相同 |
| [vllm_learn_remote_inventory.json](vllm_learn_remote_inventory.json) | 68 份副本清单、源码差异、扫描范围及权限错误 |
| [vllm_learn_unmatched_source.tar.gz](vllm_learn_unmatched_source.tar.gz) | 未按原始 SHA256 命中 Git 历史的实验源码和 4 份未提交旧 v6 文件；不作为运行入口 |
| [release_audit](release_audit/) | 发布前补跑的默认/NCCL 回归日志和退出状态 |

最终源码在服务器 `/data/ifly_jyhe23/vllm-learn-mm-benchmark-v2-20261108`，本地实现位于 `C:/Users/27363/AppData/Local/Temp/opencode/vllm-learn`。两处 107 个 v9 Python 文件已逐项核对相同。本轮 GPU 实验和 HTTP 服务均已结束。修复及仓库文档已提交为 `b3bed86`（分支 `fix/mm-graph-validation`）。核验记录见 [release_receipt.json](release_receipt.json)。

目录内其它 `verified_intermediate*`、`graph_verified*`、`final_graph_evidence*`、`quality_256_evidence*`、旧 `v9_verified_source.zip` 和旧 `source_manifest.json` 为阶段产物，用于追溯。它们不代替上表的最终源码与证据。

发布整理记录见 [publish_audit.json](publish_audit.json)，本目录全部文件的大小/SHA256 见 [publication_manifest.json](publication_manifest.json)。GitHub 发布提交与实现基线提交不同，差异为文档、脚本、证据和字节保全属性。
