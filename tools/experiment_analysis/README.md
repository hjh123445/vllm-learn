# 实验分析脚本

这些脚本保留实际诊断过程，和 `v9/tests/` 的采集/验证入口配合使用；不是在线服务的运行依赖。请从仓库根目录启动，使用与报告一致的 PyTorch 环境。

- `analyze_layer_activations.py`：对照已保存的 HF/自研层级张量。
- `analyze_numerical_diagnostics.py`：完整 logits 的误差、argmax、tie 和 margin 对照。
- `analyze_cached_teacher_056.py`、`analyze_batch_controls_056.py`：固定前缀与匹配 batch 执行路径分析。
- `probe_natural_generation_056.py`：单请求/批处理自然生成的首次分歧比较。
- `analyze_verified_traces.py`：GPU 时间区间与热点分析。
- `build_custom_ep8_compact_trace.py`：从八个原始 rank trace 选取首条消息附近窗口，保留真实事件并补充轨道/层标注。默认读取根目录 `outputs/verified_parallel/rollout_trace_056_20261104_152543/trace_ep8/trace`；这些大型原始文件不包含在公开 Git 目录，需要先从原实验归档恢复。
- `audit_trace_coverage_056.py`：原远端覆盖检查脚本，保留原主机断言与绝对路径，只能在对应实验机/路径执行；不是通用本地工具。

`.pt` 张量及原始大 trace 未随 Git 发布；元数据中的绝对路径是历史运行记录，重新分析前需映射到实际归档路径。不能把无法读取原始数据、空比较结果或未跑完的脚本视为验证通过。
