# 模型侧Profiler清理验证

2026-11-05。移除模型/层内的record_function、手工逐层CPU计时、专家计数字段与导出方法；ModelRunner的重复范围也已移除。模块范围由 `v9/utils/module_profiling.py` 在采集模式注册可移除hook。worker仍管理CPU/CUDA采集及PP时间线；不在正常推理安装模块profiling hook。

本轮在172.28.30.56隔离源码快照中进行CPU/Gloo回归，显式隐藏CUDA设备。**28 passed in 16.82s**。没有GPU性能实验、30B权重加载或新的CUDA trace。

验证内容：真实CPU profiler记录层/Attention/MoE/router范围及ATen；嵌套父子关系正确；启用hook前后输出相同；异常后范围闭合且hook清除；不删除其他观察者的hook；现有MoE DP×TP全专家oracle、PP实际tiny模型、并行权重组件、batch策略和trace合并。

`remote_command.txt` 是实际命令；`status.json` 记录机器、隔离路径与范围；`stdout.log`、`stderr.log`、`results.xml` 是原始结果；`validation_sources.tar.gz` 与 `source_sha256.json` 固定该轮源码。远端tar的future timestamp提示来自机器时钟差异，不是测试失败。测试后仅修正模型文件末尾空行并补充本文档。

层内EP dispatch/expert/combine是内部方法，当前模块hook不提供这些独立语义范围。CUDA/NCCL活动仍由底层profiler采集，但本轮未重新验收GPU采集。旧专家统计入口及 `--expert-counters` 已移除；旧历史trace/数据不修改。先前BF16自由生成分叉仍是独立未解决问题，本次测试不能据此宣称已修复。
