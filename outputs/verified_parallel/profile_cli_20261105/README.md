# Profile与Bench CLI验收

2026-11-05。本轮使用172.28.30.56隔离源码快照、自生成tiny模型/输入，不加载30B，不把smoke吞吐当项目性能。没有推送GitHub。命令说明：仓库v9/BENCH_PROFILE_CLI.md。

- cpu：85项通过，25.47秒，含真实loopback HTTP、SSE usage计数、CLI help、profile数据隔离和既有CPU/Gloo回归。
- gpu：首轮2项失败，有ATen/模块标注而无CUDA runtime/kernel。原始日志保留。独立torch矩阵乘法也复现；CUPTI直接活动接口有数据，显式flush可恢复Kineto导出。没有把失败当成功。
- gpu_flush：兼容修复后2项通过，44.89秒。真实生产worker TP2及PP2TP2 EP路径，分别运行无profile与profile，生成token一致。逐rank检查ATen、CUDA runtime、GPU/NCCL kernel和层标签。
- gpu_cli_e2e：1项通过，23.78秒。真实GPU离线bench/profile CLI和单卡HTTP profile服务/客户端CLI端到端，检查usage与profile性能指标抑制。

PP2TP2 trace各rank GPU kernel数1186/1167/1226/1195，NCCL各64；每rank可见stream7及31。TP2两rank kernel数2787/2736，NCCL各116。数据只代表这些短合成窗口。

合并文件 `pp2tp2_ep2_cuda_trace.json.gz` 已实际用Perfetto trace processor解析：4774个kernel slice、9777个CUDA runtime slice、20379个cpu_op slice、5443条flow，解析错误统计为空。`perfetto_validation.json`保留查询结果。真实stream轨道存在，不是只有record_function。

每轮目录保存命令、status、stdout/stderr、JUnit、源码快照及SHA；GPU原始trace与输入/输出在trace_artifacts.tar.gz，成功轮已解压到artifacts（跳过pytest绝对路径符号链接）。CUPTI调试日志在gpu目录。兼容代码只加载本机已有匹配CUDA库并flush；另行试装的CUPTI13.0.85位于隔离/tmp目录，未替换系统环境，升级本身未解决问题。

限制：未验收30B/HF全负载正确性；之前BF16自由生成分叉尚未解决。HTTP多DP、动态start/stop profile API、Poisson开环及SLO goodput未实现。新CLI默认closed-loop，profile与bench分开。P95/P99来自实际观测次序统计，少样本不能声称稳定尾延迟。合成trace中的计算/NCCL交集不代表生产重叠收益。
