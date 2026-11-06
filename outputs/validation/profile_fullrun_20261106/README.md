# --profile 全程采集与完整性修复验证

2026-11-06，同机8×A800 80GB；Python3.10.21、Torch2.9.1+cu130、CUDA13.0、Triton3.5.1。只使用自生成小模型和合成请求，未读取业务/许可不明数据。

## 先打开这两个文件

- [八卡完整合并 trace](tiny_pp2ep4/trace.json.gz)：导入 Perfetto，展开各 rank 的 CPU 与 GPU/CUDA stream，多卡共用时间轴。
- [八卡概览](tiny_pp2ep4/analysis/timeline.html) / [分析](tiny_pp2ep4/analysis/report.md)：概览只供定位，精确时长看完整 trace。

**本次验证采集工具与调用链，不是 Qwen3-Omni-30B 的吞吐或质量成绩。**

## 具体修复了什么

1. 默认从第一条真实 forward 采到正常关闭，`profile_messages=0` 表示不限消息数；不强制修改 workloads/repeats。正常 serve/bench 只需额外加 `--profile`，请求/并发/batch 上限照常设置。
2. 旧验收只检查文件有 kernel，可能漏掉“某条消息完全没有 GPU 活动”。现在保存独立的执行消息清单和本 stage 层区间，逐消息核对 CPU/GPU Attention+MoE 覆盖，失败 forward/整条缺失/尾层缺失/launch 缺 GPU 关联均失败。
3. stop 前同步设备并刷新 CUPTI，trace 先写 partial 再原子改名。同步只发生在采样边界，未在每层/每消息插入同步。
4. 原 EngineCore 关闭只等 worker5秒，可能在大 trace 导出/分析时杀掉它。改用统一 worker timeout 清理预算并核对退出码；AsyncEngine owner thread 同样等待该清理预算。这是通用生命周期修复，engine 不引入 profiler 依赖。
5. profile 输出目录拒绝混旧 trace；普通 server 只加 --profile 时自动选择唯一目录。多卡合并继续保留完整事件与 CPU→GPU flow，不串接各卡时间。
6. 独立工具与引擎隔离仍由自动测试检查。新增的直接运行 trace_check CLI 测试发现缺少 main 入口、静默退出，已补上入口并重新通过17项工具回归。

## 怎么跑的

精确远端命令：[remote_command.txt](remote_command.txt)。复现入口是普通 pytest，本次设置 `PP_TEST_BACKEND=nccl`，实际运行 `test_cuda_profile_runtime.py` 中的 spawned worker 和 HTTP 子进程。

主回归 **41 passed / 141.71s**，包括：

- TP2+EP2、PP2×TP2+EP2、PP2×TP4+EP4：同拓扑分别关闭/开启 profile，贪心生成 token 数组相等，首 prefill 至末 decode 均覆盖。
- 八卡：8层、hidden16、4专家自生成模型；3请求，输入 token IDs 为 `[1,2]`、`[3,4,5]`、`[1,2]`，各生成8token；微批预算8、最大在途2。配置/输入见 [model_config.json](tiny_pp2ep4/model_config.json) 和 [inputs.json](tiny_pp2ep4/inputs.json)。
- offline CLI：两个负载各两轮、max_batch_size=4；无 profile 输出速率和P50/P95/P99；有限窗口仅采2消息；默认全程模式记录全部执行。结果见 `cli/offline_*/`。
- HTTP CLI：只加 --profile，自动创建目录；8请求/4并发，关闭后全程验收通过。无 profile-dir 也能使用，HTTP 客户端通过 health 自动判为诊断运行，速率为 null。
- 空 DP worker 只要求实际执行的MoE；注入 GPU尾层缺失、整消息GPU缺失、整CPU消息缺失、CPU尾层缺失、清单缺口、缺manifest、重复rank文件必须被拒绝。

随后对最新源码补验：HTTP及多负载bench通过；新直接运行验收CLI的测试曾失败（返回0但没执行main，stdout为空），修正后 **17项工具测试全部通过**，最终本地直接执行八卡验收命令也通过。完整失败与修正记录保留在 `cli_followup/`、`tools_followup/`，不掩盖失败。

主回归源码指纹见 `source_sha256.json`，最后源码指纹见 `final_source_sha256.json`；两者差异为 server 启动时打印路径、trace_check 独立入口及对应测试，详见 `validation_scope.json`。不把后续变化伪称为主回归原快照。

## 八卡逐消息验收

每条消息在本卡4层上都需要 Attention+MoE 共8个 CPU/GPU 组件；每卡16消息，首条为prefill、末条为decode。原始清单、压缩trace和最终验收在 `tiny_pp2ep4/raw/`。

|Rank|Stage|层区间（右开）|消息数|CUDA kernel|NCCL kernel|逐消息覆盖|
|---:|---:|---|---:|---:|---:|---|
|0|0|[0, 4]|16|8941|496|通过|
|1|0|[0, 4]|16|8462|496|通过|
|2|0|[0, 4]|16|7824|496|通过|
|3|0|[0, 4]|16|7590|496|通过|
|4|1|[4, 8]|16|9254|496|通过|
|5|1|[4, 8]|16|8696|496|通过|
|6|1|[4, 8]|16|8214|496|通过|
|7|1|[4, 8]|16|7999|496|通过|

八卡共 **66,980个CUDA kernel，其中3,968个NCCL kernel**。完整合并保留785,258个事件。GPU层归属通过launch correlation/External id建立，不凭CPU/GPU在横轴上的位置猜测。

这个小模型 trace 中，各卡计算与NCCL的GPU交集均为0；跨stage的非NCCL计算交集为10.766ms。这说明本样例观察到跨stage流水，而没有观察到同卡计算通信交叠。不能据此推广到30B重叠率或性能收益，也不能把整体空白率直接叫PP气泡。stderr中仍有PyTorch关于该ProcessGroup未batch的P2P可能序列化的警告；本轮未把这个潜在性能问题包装成已解决。

## 旧 rank5 缺尾怎样被重新检出

原文件保持不改，重新分析结果：[legacy_rank5_audit.json](legacy_rank5_audit.json)。去掉历史嵌套lane标签的重复别名后，共4条canonical消息：

|消息|GPU kernel|GPU层组件数（完整应96）|判断|
|---|---:|---:|---|
|message1 prefill|10,358|96|已有完整组件覆盖|
|message2 decode|9,060|96|已有完整组件覆盖|
|message3 decode|6,571|69|从layer34.moe起缺GPU归属|
|message4 decode|0|0|整个消息无GPU活动记录|

旧 manifest 缺独立执行清单/层区间，也不能认证。以上是**采样/关联缺口证据**，不能推出模型少算了这些层；丢失活动不能靠merge补出，必须重新采集。

## 文件与边界

`results.xml`、`stdout.log`、`stderr.log` 是主回归原始记录；`validation_sources.tar.gz` 是主回归源码快照。`environment.json`、前后资源记录用于确认环境。小模型完整测试产物另在本地工作归档保留，仓库仅保留这组八卡原始与合并trace、必要CLI结果、清单及证据，避免重复上传所有测试trace。

采的是每个GPU worker的CPU/ATen/CUDA/NCCL，**不包含HTTP/tokenizer/父调度主线程、初始化或媒体encoder**。`cuda_verified`只说明有kernel；`completeness_verified`表示逐消息覆盖及launch检查通过，不是证明任意事件绝不丢失。没有在这里宣称已验证整模CUDA Graph采集。

正常使用及分析命令见 [PROFILE.md](../../../docs/PROFILE.md)。profile开销会影响调度时间，真实性能请另跑无 --profile 的相同输入，不使用这组小模型timing得出性能优化百分比。
