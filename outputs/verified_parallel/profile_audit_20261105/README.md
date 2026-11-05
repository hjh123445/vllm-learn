# 本次 profiling 审计产物

2026-11-05。本目录全部来自已有数据的离线处理，没有新跑GPU实验。

## 打开哪个文件

用 https://ui.perfetto.dev/ 的 Open trace file 打开：

`pp2ep4_two_stages_full_cpu_cuda.json.gz`

文件包含旧采集的rank0/stage0与rank4/stage1，各自CPU线程和GPU stream 7/22/23。不是一个线程，也不是完整八卡文件。原始ATen、CUDA API、GPU kernel、memcpy和flow均保留。完整阅读方法见仓库 `v9/PROFILING_GUIDE.md`。

## 原始采集条件

- 数据来源：`rollout_trace_056_20261104_152543/trace_pp2ep4_baseline`。
- Qwen3-Omni-30B-A3B-Instruct Thinker文本路径，8×A800，PP2×EP4，旧版baseline调度。
- 8个自生成输入，每个实际47个prompt token；采集脚本限制生成4个token。inputs里的max_tokens=32不是本次实际生成上限。
- token budget 512、旧microbatch token参数128、KV上限128块，eager，prefix cache关闭。
- CPU+CUDA torch profiler；record_shapes开启、stack/memory关闭；从worker message1开始请求最多8条message，实际消息覆盖要以捕获记录为准。
- 模型初始化不在本窗口；窗口边界同步与导出会改变调度；不能用它计量生产吞吐。
- 本次只选两份源文件，不能据此说明另六个rank的状态。

新V1 PP队列是在这次历史采集之后修改的；本文件不能证明新实现性能或重叠情况。

## 如何重新合并

在仓库根目录运行。`SOURCE_DIR` 指向已有 `trace_catalog_20261105` 原始压缩文件目录，Linux/Windows均可用Python执行：

```bash
python v9/tests/merge_pipeline_trace.py --same-host \
  --inputs "$SOURCE_DIR/pp2ep4_baseline_stage0_rank0_full_cpu_cuda.json.gz" \
           "$SOURCE_DIR/pp2ep4_baseline_stage1_rank4_full_cpu_cuda.json.gz" \
  --output outputs/verified_parallel/profile_audit_20261105/pp2ep4_two_stages_full_cpu_cuda.json.gz
```

本机使用的SOURCE_DIR：
`C:/Users/27363/Documents/Codex/2026-11-01/qing/outputs/verified_parallel/trace_catalog_20261105`。

脚本按Kineto base校正，只减一个共同起点；各rank没有分别归零，也没有顺序拼接。

## 各文件用途

| 文件 | 用途 |
|---|---|
| pp2ep4_two_stages_full_cpu_cuda.json.gz | 可打开的多轨道CPU+CUDA trace |
| perfetto_validation.json | 官方Perfetto v59.1实际解析的轨道、事件、flow及错误检查 |
| merge_audit.json | 输入SHA、时间变换、事件数量、GPU区间并集/交集、离线测试结果 |
| kernel_oracle_audit.json | 使用匹配cached HF terminal-head参考，重算历史Triton消融的逐步token匹配 |

Perfetto解析到64,331个kernel、498,482个cpu_op、7,621个GPU memcpy和71,982条flow，解析错误0。两GPU非NCCL kernel有约11.77ms交集；各GPU自身的NCCL与非NCCL kernel交集为0。这是短profile窗口统计，不是稳态利用率、PP bubble ratio或新性能收益。

Triton消融是历史12个合成prompt×4个位置的teacher forcing，不是任务准确率。直接使用旧requests里的expected_next_token字段和使用匹配cached HF raw_argmax_ids，是两种不同参考；审计文件把两者分开记录，避免混淆。
