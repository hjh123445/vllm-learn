# PP V1 风格改造验收记录

日期：2026-11-05。实现说明见 [PP_V1_IMPLEMENTATION.md](../../../v9/PP_V1_IMPLEMENTATION.md)。本目录是本次新实现的验证，不能与旧 PP trace、旧 tok/s 混用。

| 验证 | 环境 | 结果 |
|---|---|---|
| scheduler/KV、持久队列、真实 spawn/SHM、在线适配、worker health、TP 组件、CPU 模型集成 | 172.28.30.56，Python 3.10.21，PyTorch 2.9.1+cu130，CPU/Gloo | **43 passed，11.96 s** |
| 真实 Qwen MoE 模型代码＋ModelRunner＋新 executor，FP32 串行参考与 PP4、PP2×EP2、PP2×TP2 | 同机 GPU 4/5/6/7，NCCL，合成小权重 | **3 passed** |
| 生产 EngineCore/worker_main：初始化、真实 loader、warmup、BF16 缓存生成、取消、正常退出 | 同上，临时合成 checkpoint/tokenizer；单卡和三种四卡拓扑 | **1 passed** |
| 上述两组 NCCL 测试合计 | 同上 | **4 passed，56.74 s** |

每组的 `status.json` 描述环境和范围；`remote_command.txt` 保留精确执行命令；`remote_stdout.log`、`remote_stderr.log`、`results.xml` 保留原始日志和逐测试结果；`source_sha256.json` 和 `validation_sources.tar.gz` 保留该次源码快照。最终 cpu/nccl 快照包含完全相同的 Python 源码。根目录另保留了第一次 CPU 测试快照，属于中间版本；最终结论以 cpu/、nccl/ 为准。

测试 subprocess 正常返回，生产 worker 测试要求所有 worker exitcode=0；没有后台持续实验，没有使用公开许可不明数据，没有修改远程正式 checkout，没有推送 GitHub。

## 核心回归证据

- 首次 `step` 只提交时无输出，后续队列仍保留该 Future。
- 队首 batch 回收后，其请求进入下一批；尾部 batch 的 Future 仍未完成，验证没有整轮 drain barrier。
- 阻止末 stage 接收首条控制消息时，首 stage 已接收到第 4 条，验证没有全 worker ACK 屏障。
- 8 条在途 RPC 通过真实单槽 SHM 输出，Core 不逐条读结果也不会卡住，因为 executor 接收线程独立 ACK。
- 对在途请求 abort 后，KV 保留到匹配结果回收；该输出被丢弃，不错误写入用户流。
- 在途请求不能被重复调度/抢占；未完成 prefix 不进入共享 hash。
- 错误 batch ID、seq_ids、输出数量、worker 退出都作为错误传播。
- 正常 shutdown 中前级 worker 先退出，不会被后级 shutdown ACK 等待误判为故障。
- 实际生产入口的单卡、PP4、PP2×EP2、PP2×TP2 合成 BF16 缓存生成输出一致。

## 可复现命令

在具备项目依赖的 Linux 环境，从仓库根目录执行：

```bash
PYTHONPATH=v9 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
python -m pytest -q \
  v9/tests/test_pipeline_batch_queue.py \
  v9/tests/test_pipeline_executor.py \
  v9/tests/test_scheduler_preemption.py \
  v9/tests/test_pipeline_schedule.py \
  v9/tests/test_pipeline_schedule_topology.py \
  v9/tests/test_async_engine.py \
  v9/tests/test_worker_health.py \
  v9/tests/test_model_runner_batch_policy.py \
  v9/tests/test_parallel_components.py \
  v9/tests/test_pipeline_model_runtime.py
```

明确选择四张可用 GPU 后：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PP_TEST_BACKEND=nccl \
PYTHONPATH=v9 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
python -m pytest -q \
  v9/tests/test_pipeline_model_runtime.py \
  v9/tests/test_pipeline_worker_runtime.py
```

不设置 `PP_TEST_BACKEND=nccl` 时，生产入口测试明确 skip，不会悄悄启动 GPU。远程本次在独立 `/tmp/vllm-learn-pp-v1-*` 快照安装 pytest 到局部 testdeps；stderr 中 pip 对已有 PyOpenSSL/cryptography 兼容性的提示不是 pytest 失败，没有改服务器 base Python 包。本地/远端时钟偏差也产生 tar future timestamp 提示，测试结果和源码内容不受影响。

## 本次没有得到的结论

- 没有测 Qwen3-Omni-30B 新版的吞吐、TTFT、TPOT、p99。
- 没有重新采集真实 30B CUDA trace，不宣称已经证明通信重叠或气泡减少多少。
- 小型合成模型的 token 一致性不等于完整 30B 的 HF 精度认证。
- 仍没有同请求未知输出 token 的异步反馈、lazy receive accessor、同卡多 stage 或训练 DualPipe。

本次完成的是运行时结构与正确性改造；后续性能实验应固定模型/输入/数值策略，比较 `max_concurrent_batches=1` 与 PP 深度，独立采集 profile 和无 profile 指标。
