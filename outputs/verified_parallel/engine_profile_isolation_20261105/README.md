# Engine与观测工具隔离验证

2026-11-05，172.28.30.56隔离临时目录，显式隐藏CUDA。**77 passed in 24.21s**。本轮未加载30B、未进行GPU实验或吞吐测量，未采集新CUDA trace。

## 改动

- engine/worker不再负责profiler生命周期、record_function、trace列表、显存导出。
- ModelRunner不再读取logits/activation采集配置，不再导出诊断张量。
- PipelineTransport不再计时或积累trace；未使用的PipelineTrace/TraceEvent类删除。
- 通用runner_cls/transport_cls允许实验脚本显式选择utils/observed_runner中的实现。默认engine不导入观测工具，仅继承采集环境变量不会触发采集。
- 独立观测Runner在sampler前用hook保存logits，管理模块hook、profiling窗口和导出；观测Transport记录CPU调用区间。没有全局替换模型、engine或torch通信函数。

KV预算、请求完成/服务延迟指标、超时检测和通信正确性所需的CUDA事件/同步保留在运行代码内。

## 验证

77项包含现有调度/KV、安全退出、PP队列/executor、真实tiny模型PP、DP×TP专家oracle、trace合并与模块范围测试，以及新增观测边界测试：

- 普通EngineCore忽略诊断环境变量；显式选择观测Runner才保存张量，warmup不进入采集。
- 真实CPU profiler仅采指定消息窗口，保留ATen和层/Attention/MoE语义，结束后移除hook并生成窗口元数据。
- 逐请求诊断不会混淆请求行，独立DP副本目录隔离，shutdown清除采样hook。
- 两进程真实Gloo分别验证同步/异步观测Transport，6批不同形状、双lane、credit=1，传输内容和batch标识正确。
- 源码边界检查确认engine/models/layers没有profiler、采集环境变量或观测工具依赖。

stdout.log/stderr.log/results.xml保存原始结果；remote_command.txt记录实际命令；status.json记录机器、范围与临时路径；validation_sources.tar.gz和source_sha256.json固定测试源码。tar的future timestamp提示来自本机与远端时钟差，不是测试失败。测试后仅添加通用工厂注释和文档。

## 限制

观测Transport的异步send区间现在是send_forward的CPU调用时间，与历史从发送到完成等待的手工范围定义不同。不能用其时长推断GPU传输或计算/通信重叠。新采集架构尚未进行NCCL/CUDA trace回归；CPU通过不证明GPU profiler完整性。先前BF16严格自由生成分叉仍未解决，本次没有把它当作通过。
