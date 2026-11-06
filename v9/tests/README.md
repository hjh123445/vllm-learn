# tests：自动回归，不放实验脚本

旧硬编码bench、通信演示、capture/merge/probe入口已移除或迁往 `../tools/`。保留测试保护当前执行路径，正常使用引擎不必逐个阅读。未接入运行时的pipeline planner及其测试已删除；快照契约并入 `pipeline_batch_queue`。

| 关注点 | 对应测试（去掉共同的 `test_` 前缀） |
|---|---|
| Profile正确性 | `profile_tools`、`trace_merge`、`module_profiling`、`observed_runner`、`cuda_profile_runtime` |
| Bench/HTTP/异步接口 | `bench_cli`、`http_server`、`async_engine` |
| PP调度与IPC | `pipeline_batch_queue`、`pipeline_executor`、`pipeline_links` |
| PP真实模型与生产worker | `pipeline_model_runtime`、`pipeline_worker_runtime` |
| EP/TP/DP | `coupled_moe`、`distributed_moe`、`ep_topology`、`ep_token_ownership`、`parallel_components`、`dp_frontend` |
| 数值与kernel | `attention_reference`、`rotary_reference`、`route_reduction`、`silu_and_mul`、`quantized_fallback`、`auto_grouped_backend` |
| 原 cached 精度反例与离线证据 | `pipeline_worker_runtime` 中 FP32 专家对照、`cached_logits`；过程见 [精度排查](../../docs/精度排查.md) |
| 调度、KV、观测与健康 | `scheduler_preemption`、`prefix_cache_switch`、`model_runner_batch_policy`、`activation_capture`、`worker_health` |

只验证标准库trace工具，不需要GPU或PyTorch：

```bash
python -m pytest v9/tests/test_trace_merge.py v9/tests/test_profile_tools.py -q
```

仅工具回归：

```bash
python -m pytest v9/tests/test_trace_merge.py v9/tests/test_profile_tools.py \
  v9/tests/test_bench_cli.py v9/tests/test_module_profiling.py \
  v9/tests/test_observed_runner.py v9/tests/test_silu_and_mul.py -q
```

该选择包括CPU契约、loopback HTTP和CUDA SiLU测试。2026-11-06运行时清理时的完整pytest：109 passed、8 skipped、12 subtests passed（先于后续精度工具与策略）。另行显式启动NCCL测试得到10 passed、1 failed；失败为DP2×TP2开启EP后的BF16严格greedy不一致，清理前版本复跑同样失败。没有修改断言掩盖问题。原始命令、源码指纹与日志见 [验收记录](../../outputs/validation/runtime_cleanup_20261106/final/)。

后续精度排查：仅专家FP32的原生产生成/单侧idle-DP回归，加上三种DP×TP和两种精度的组件oracle，共7 passed；默认精度的原三种生产拓扑仍为2 passed、1 failed；相关单元回归20 passed、12 subtests passed；离线分析2 passed。默认BF16失败没有放宽断言，新策略的通过不代表默认已通过。完整原始张量与复现见 [本轮精度证据](../../outputs/correctness/ep_diagnosis_20261106/README.md)。

`pipeline_worker_runtime` 默认跳过，显式 `PP_TEST_BACKEND=nccl` 才启动真实多卡进程。其他分布式测试也有自己的opt-in环境约束，先看文件头；不要误把skip算成已通过GPU验证。fixture生成随机小权重，不下载真实模型、不接触未知数据集。

`rotary_reference` 现在使用合成配置与HF比较，不再在模块导入时把pytest命令行参数当作权重路径。`pipeline_links`、`distributed_moe`、`auto_grouped_backend` 是独立CLI oracle，普通pytest不会自动执行它们的全部通信/微基准逻辑；`pipeline_links` 已另行以四进程Gloo、NCCL运行，日志在上级验收目录。
