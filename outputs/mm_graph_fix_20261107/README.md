# 本轮证据索引

先阅读仓库 docs/CUDA_Graph补齐精度排查.md 和 docs/多模态与Graph修复说明.md。主验收：verified_summary.json。

- graph_batch_invariant_kv_checked/rank0..7.json：最终严格验证，256个rank-step，未补齐/补齐/Graph完整logits、KV及同一步重排均逐位一致，含源码SHA256。
- padding_invariant_final/padding_diagnosis.json：修复后逐层无差异，dummy改值无影响，非有效slot KV不变。
- graph_invariant_cli_profile/trace.json.gz：真实八卡合并CPU/CUDA/NCCL轨迹，直接拖入Perfetto。仅TP8/EP8、PP1，不能用于宣称PP流水线已验证。
- graph_invariant_cli_profile/trace/cuda_rankN.json.gz：单卡原始轨迹；capture/audit/trace_validation是采集清单与覆盖校验。Graph replay内没有逐层Python hook，不等于每个Graph节点已有layer归属。
- graph_invariant_cli_profile/execution_rankN.json：每rank捕获bucket与实际replay计数；correctness.json仅是重复生成一致性，不是HF质量。
- padding_bf16_3to4_final/：原BF16归约形状敏感复现。padding_fp32_7to8_final/：FP32归约不能消除本地BF16 GEMM形状敏感的反例。
- graph_batch_invariant_v1、v2、padding_invariant_expert_inputs及对应log：失败和中间消融，不能算成功。
- grouped_review/：历史+8.88% grouped GEMM实验的两轮原始结果。不是本轮严格Graph模式吞吐。
- vision*、mm_chunk32、mm_full：图片组件对拍与双图执行证据；图文整模HF质量未认证。
- cpu_final.log：144 passed、10 skipped、12 subtests passed。run_environment.json：环境/源码指纹。manifest.json：本地文件校验。

其它graph_exact/graph_core等为调查早期版本，不能与最终严格模式混用结论。所有profile采集时延只作诊断，不作吞吐成绩。较大的trace保留本地；Git提交保留源码、文档和精简结果，不自动上传全部原始轨迹。
