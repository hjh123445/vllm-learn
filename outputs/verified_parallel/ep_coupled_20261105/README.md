# EP覆盖DP×TP：验证记录

日期：2026-11-05。实现说明见仓库 `v9/EP_DP_TP_IMPLEMENTATION.md`。

## 结论与范围

本轮在172.28.30.56的隔离 `/tmp/vllm-learn-ep-coupled-*` 源码快照中验证。只使用合成tiny Qwen/MoE权重和固定输入；没有加载Qwen3-Omni-30B，没有吞吐优化结论，没有使用来源不明的数据。没有覆盖远端正式工作仓库，没有推送GitHub。

| 目录 | 结果 | 解释 |
|---|---|---|
| cpu | 59 passed / 18.01s | 初版CPU/Gloo回归 |
| cpu_final | 59 passed / 17.96s | 最终功能源码CPU/Gloo回归，含诊断副本编号修复 |
| nccl | 9 passed, 1 failed / 121.78s | 分布式oracle及部分生产拓扑通过；DP2TP2 BF16自由生成与单卡分叉 |
| nccl_matched | 9 passed, 1 failed / 153.89s | 改为同DP/TP/PP配置、关闭EP的基线后仍分叉，尚未验收严格生成一致性 |
| nccl_forced | 失败 | 诊断文件覆盖导致预期16个向量实际只找到8个；保留原始失败 |
| nccl_forced_final | 1 passed / 31.15s | 修复覆盖后，16个固定前缀logits向量在指定容差内一致 |

严格自由生成的具体差异：EP `[54,60,54,60]`，基线 `[54,60,57,60]`。根因尚未证明，不能用后面的容差测试宣称这一问题已解决。

固定前缀测试：DP2TP2 EP4对比DP2TP2专家TP；最大绝对logits误差 **0.013671875**，argmax差异 **0/16**，`atol=0.02, rtol=0.03`。固定相同前缀能避免生成分叉改变后续输入，但当前前缀集合并未完整定位自由生成分叉处的首个数值偏差。因此只记为有限范围的数值检查。

## 如何复现与审计

每个目录保存：

- `remote_command.txt`：实际执行的完整远端命令，包括环境变量和测试选择。
- `status.json`：机器、隔离路径、退出码、GPU选择和范围。
- `validation_sources.tar.gz`、`source_sha256.json`：该轮真实Python源码快照及每文件指纹，避免用当前工作区倒推历史代码。
- `remote_stdout.log`、`remote_stderr.log`、`results.xml`：原始日志与逐测试结果。

在具有PyTorch、pytest及必要依赖的环境解压该轮源码，按 `remote_command.txt` 中的PYTHONPATH、环境变量和测试列表运行。命令中的解释器、临时目录和GPU编号应按本机替换。GPU测试包含多进程启动，应从脚本/pytest入口运行。

主要测试文件：

- `v9/tests/test_ep_topology.py`：拓扑、rank公式、拒绝独立EP、空闲DP驱动。
- `v9/tests/test_coupled_moe.py`：DP1TP4、DP2TP2、DP4TP1；不等长和空输入；全专家oracle；route数量守恒。
- `v9/tests/test_pipeline_model_runtime.py`：真实tiny模型PP及TP/EP组合。
- `v9/tests/test_pipeline_worker_runtime.py`：生产EngineCore/frontend/worker、BF16生成、固定前缀logits对照。

`nccl_forced_final/fixed_prefix_logits.tar.gz` 保留24个原始诊断张量；SHA256：`abedd17b2e926071f9b55b3044e76d69ed04eba43b30c48115552ac9b3ed5337`。来源路径和文件数见同目录 `capture_manifest.json`。张量为本轮自生成测试数据，仅应从可信源码快照中加载。

## 尚未验收

严格BF16自由生成一致性、真实30B/HF数值、多DP在线HTTP、异步DP协调、吞吐和通信重叠。历史独立EP实验不能直接作为本次DP×TP耦合实现的数据。

最后的worker诊断/trace文件命名和trace合并rank优先级修改晚于上述GPU源码快照；9项本地trace合并测试通过。它们未重新采集GPU trace；每轮确切代码以各自指纹为准。全v9 Python语法检查及git diff空白检查另行通过。
