# CUDA trace只有CPU事件：原因、证据与修复边界

2026-11-05。环境为172.28.30.56上的PyTorch2.9.1+cu130/CUDA13.0。本文描述该环境的可复现现象，不把它扩大为所有CUDA13安装的通用缺陷。

## 记录如何进入trace

CPU侧的ATen和record_function由PyTorch记录；GPU kernel、CUDA API、memcpy等活动由CUPTI采集到活动缓冲区，随后经回调交给Kineto处理，最后导出Chrome JSON。两条路径相关但不同，所以CPU记录正常不代表GPU活动已经交付给导出器。torch.cuda.synchronize只保证GPU工作完成，不保证CUPTI活动缓冲区已经刷新给Kineto。

## 我们实际证明了什么

1. TP2、PP2TP2采集都有ATen和层标注，却没有CUDA runtime/kernel。启用CUDA activity的参数正确，不是单纯缺少with_modules。
2. 脱离项目、单进程独立矩阵乘法torch.profiler也复现，因此不能把问题归因于模型hook或worker进程设计。supported_activities包含CUDA也不足以证明实际记录成功。
3. CUPTI subscribe返回SUCCESS；直接注册CUPTI活动buffer回调并执行GPU工作，得到9600字节有效buffer。GPU活动采集链路具有工作能力。
4. 换系统CUPTI13库、隔离安装13.0.85库、延长到10秒窗口均未解决。没有据此覆盖系统环境。
5. 同一最小PyTorch案例在GPU同步后、profiler停止前显式调用cuptiActivityFlushAll(1)，恢复10个kernel和54个CUDA runtime事件；Kineto日志从GPU trace is empty变为处理GPU records。
6. 把相同处理仅放进独立观测工具后，生产spawn worker的TP2和PP2TP2均通过，逐rank有实际GPU/NCCL记录，profile前后生成token一致。

因此可操作的原因定位是：该环境中GPU活动buffer到Kineto停止/导出之间存在刷新或交付问题。尚未对PyTorch/Kineto/CUPTI源码做二分，也没有确定具体上游缺陷号；不能断言内存buffer为何没有被默认stop路径刷新，更不能声称NCCL没有执行或GPU计算没有发生。

## 为什么修复有效

停止前先同步设备，避免活动仍在执行；随后cuptiActivityFlushAll(1)要求交付缓冲记录，Kineto再停止并导出。标志1是CUPTI_ACTIVITY_FLAG_FLUSH_FORCED，可以包含未完成记录，因此同步很重要。此处理是对当前环境的兼容方案，不是所有环境都必需的标准步骤，也不证明所有记录在任意负载下绝不丢失。

实现位于utils/cupti_flush.py，由独立观测Runner调用，默认auto只在CUDA13.0启用；只加载现有匹配CUDA major的库，不替换系统库。可用VLLM_LEARN_CUPTI_FLUSH=off禁用，on显式启用。engine与模型不导入它，无profile时不执行该同步/flush。

每次仍需检查实际活动：bench.py trace-check --require-cuda --require-nccl。有ATen、层号、文件名cuda或cuda_available都不够。该检查也不是丢事件的全面证明；还需看所有rank、窗口覆盖、correlation/flow、Perfetto解析错误与关键通信序列。

## 对吞吐与尾延迟的影响

额外同步、buffer处理和trace导出会改变调度，profile只用于解释行为。吞吐和P95/P99必须来自关闭profile的独立运行。修复采集不等于改善模型toks/s。原始失败、CUPTI调试日志、成功GPU测试与Perfetto解析记录在outputs/verified_parallel/profile_cli_20261105。
