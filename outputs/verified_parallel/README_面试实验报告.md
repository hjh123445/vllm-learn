# Qwen3-Omni 推理引擎：实验、诊断与面试报告

新增：面试简历中的动态负载并行选型与精度校验项目描述，以及对应证据边界，见[动态负载并行策略与简历证据](动态负载并行策略与简历证据.md)。

> 11月4日20:40更新：新增显式 `batch_execution_mode=per_request` 作为数值诊断/保守路径；EP8短输入batch=8、1K batch=3以及PP2×EP4短输入batch=3的选定层边界逐元素一致。默认 `batched` 仍保留融合吞吐，并在1K batch=3的L0 o_proj复现BF16 M形状差异；per_request不是默认性能修复。RoPE设备舍入与EP ownership/router形状两处实现问题已通过受控重放修复；匹配执行路径的EP8、PP2×EP4在12条合成输入×32个固定前缀上，各384/384 logits逐元素匹配HF。诊断hook原位residual污染已改用pre-hook并通过测试。旧快路径在同机vLLM基准约慢3–15倍；真实HTTP负载完成至256并发。结论与边界见[准确性修复与在线服务复核](准确性修复与在线服务复核.md)。

2026-11-04 更新。旧机夜间实验已停止；用户随后授权在 **172.28.30.56 八卡**继续验证。两台机器的数据分别保留，不混合计算性能。硬件为 **8×A800 80GB，SM80**，模型为 Qwen3-Omni-30B-A3B-Instruct。本轮不推送 GitHub。

## 结论与完成边界

EP、TP、PP×EP、DP×EP、DP×PP×EP 已进行文本推理实验。**目前没有依据声称超过vLLM，也没有实现完整DualPipe。** 项目价值在可复现的数值排查、数据所有权分析、负载驱动的并行选择和多卡CUDA trace。

关键发现有两个已定位的数值根因：CPU预计算RoPE与GPU FP32三角函数存在舍入差异；ownership先切token再算router会改变BF16 GEMM的M维。两者已修复并纳入安全默认。匹配路径384/384逐点相同，但自然批量对比仍仅5/12全文一致；新探针把一类跨batch差异复现到BF16 q_proj/o_proj的M形状舍入，尚未修复全模型跨形状变化。见[数值正确性排查](数值正确性排查.md)和[新机复核报告](056正确性与消融复核.md)。性能数字只适用于对应执行路径，不替代HF质量验证。

| 项目 | 已有证据 | 边界 |
|---|---|---|
| EP变长通信 | variable splits、空source、不同source长度、TP2×EP2 oracle通过 | NCCL实现，尚非DeepEP |
| 专家权重 | 每rank只分配E_local个专家 | Router仍需全局专家logits |
| PP | 多stage、多microbatch、有限在途消息、异步send；PP2×EP4逐请求集成捕获通过 | worker串行，recv阻塞；不是训练DualPipe |
| DP | 多进程EngineCore、副本GPU隔离、独立scheduler/KV | step聚合等慢副本；不是训练DDP |
| 初始化 | CPU线程过度订阅修复，旧410–413s降至34.55s | 初始化不混入tok/s |
| native专家后端 | A800已有PyTorch grouped API可运行、组件误差零 | 仍核查内部kernel，API名不证明融合 |
| 多模态参考 | HF图片/音频/视频processor+encoder+thinker均通过 | 自研仍只文本，不是MM serving |
| 官方vLLM对照 | 同机vLLM0.14.1与自研均完成三轮顺序基准 | 官方在所测负载明显更快；拓扑和注意力实现不同，并非严格同构算子比较 |
| 量化、投机解码 | 旧原型负收益复盘 | 未集成当前serving |

## 早期候选并行矩阵（历史数学路径）

固定BF16、eager、prefix caching关闭、上下文上限8192、token budget512、序列上限128，每项三轮。输入IDs持久化、输出长度固定；吞吐不含初始化/profiler。下表使用早期数学/快路径，尚未通过最终整模质量门槛；只用于候选筛选，不是修复后默认参考路径成绩。当前同机表见[准确性复核](准确性修复与在线服务复核.md)。

单位output tok/s，中位数：

| 8卡布局 | 短64请求 | 短128请求 | 1K输入8请求 | 4K输入8请求 | 混合64请求 |
|---|---:|---:|---:|---:|---:|
| EP8 | 410.37 | 652.86 | 38.91 | 14.07 | 89.51 |
| DP2×EP4 | 397.13 | 673.37 | 43.71 | 21.68 | 78.79 |
| DP4×EP2 | 354.29 | 606.38 | 46.05 | 28.17 | 64.69 |
| PP2×EP4 | 244.85 | 318.85 | 29.86 | 12.61 | 66.37 |
| PP4×EP2 | 174.70 | 226.17 | 23.05 | 10.01 | 50.41 |
| DP2×PP2×EP2 | 244.76 | 346.13 | 33.43 | 16.79 | 52.19 |
| TP2×EP4 | 327.70 | 506.18 | 31.29 | 12.92 | 76.32 |
| TP4×EP2 | 241.42 | 366.98 | 23.30 | 9.98 | 56.34 |
| TP8 | 179.63 | 257.52 | 17.07 | 7.10 | 38.84 |

本轮没有通吃所有负载的布局：短输入高并发EP8/DP2×EP4较好，4K小并发DP4×EP2较高，PP固定策略未胜出。应结合复制Attention、collective次数、microbatch和填充排空分析，不能只用“EP适合MoE”“PP省显存”解释。

这是特定实现的观测结果，不是策略理论上限。本项目并行轴独立，DP2×EP4是两个独立EP4池；官方vLLM的EP group构造与此不同，不等于attention DP2、专家跨8卡的同一EP8池。见[官方语义审计](官方并行语义与代码结构审计.md)。

轮次范围、TTFT/TPOT、长短请求分组和新增配置更新于[完整矩阵](新实验矩阵与尾延迟.md)与[CSV](overnight_summary.csv)。8请求P99近似最大值，不是线上SLA；arrival64按engine step到达，不是跨框架相同墙钟QPS。

## 如何看profiler

打开[离线TRACE阅读器](TRACE交互阅读器.html)，按场景、GPU、stream、层、microbatch筛选。对应的`*_gpu_timeline.json`可导入Perfetto，保留实际CUDA kernel、原kernel名、全局层号、Attention/MoE、stage、EP rank和MB/lane。

阅读器长条是kernel包络，包络内可能有空隙；busy才是区间并集。覆盖率不等于SM利用率，NCCL kernel可能等待。CPU annotation不等于GPU执行，嵌套range不能累加。

已逐rank、逐消息检查覆盖。旧trace存在缺卡/缺层；部分新窗口最后一条消息仍缺尾层归属。warm EP/PP窗口第一条被捕获消息在八卡均覆盖负责的全部层，最后一条并不完整；不能把缺失事件当气泡。见[TRACE指南](TRACE阅读指南.md)。

## 多模态做到哪里

已经加载三种合成输入：224×224红色图形图片、1秒440Hz音频、4帧合成视频。HF processor、预训练视觉/音频encoder、thinker forward均成功。自研引擎仍未接入这些媒体请求。

多模态serving还需media request schema、embedding插入/token对齐、时空MRoPE、视频时间戳/音视频对齐、encoder调度与缓存/显存预算，以及媒体预处理到LLM prefill的端到端trace。Talker音频生成是另一条输出路径。模型家族支持多模态不等于文本引擎自动支持。

## 在线服务做到哪里

已修复AsyncEngine提交与step共锁、结果漏收、完成态内存清理、无界队列、取消/关闭错误；补齐SSE结束、Unicode尾块、异常、超时和断连取消。CPU并发回归、loopback HTTP和KV抢占回归通过。真实HTTP在EP8/512、EP8/2048、PP2×EP4/512分别完成2352个请求，覆盖闭环1/8/32/128/256并发与Poisson 1/4/12rps，并验证direct、非流式和SSE结果一致。

最高256并发时，EP8/512 TTFT P99为21.26秒、EP8/2048为6.07秒、PP2×EP4/512为21.09秒。token budget扩大降低本轮观测到的排队尾延迟，但共享机单轮数据不能证明布局优劣。客户端TTFT还包括body读取、parse/template/tokenize、admission与scheduler排队、detokenize、socket写出和客户端读取；SSE chunk不能当作token。当前是loopback和合成文本负载，不代表生产网络或视频请求分布。

流量端的多模态额外成本包括上传/解码、图像/音频预处理、视觉/音频 encoder、动态特征序列长度、时空位置/时间戳处理、media cache、KV 预算及多模态输出序列化。占位符 token 本身不编码媒体内容；必须把 processor/encoder 生成的特征按模型协议注入 thinker。自研引擎仍是文本服务，尚未实现这些路径。

本轮文本和媒体均自行生成，未读取未知服务器数据集。公开数据也需核验来源、版本、许可和媒体权利。

## 面试最值得讲的三个问题

1. **重复路由与数值偏差**：追踪token owner，控制输入/route/GEMM形状，用replay与固定前缀定位，再测ownership的通信与性能代价。
2. **并行最优为何随负载改变**：固定8卡和输入/输出/精度，测试EP/PP/DP/TP混合；拆分长短请求TTFT/TPOT，解释负收益。
3. **profiler如何形成证据**：先检查CUDA和层覆盖，再看跨stage计算并发、同卡通信/计算、host同步和launch密度；不以异步API代表overlap。

讲稿与追问见[面试讲稿](面试讲稿与追问.md)，动机与可证伪假设见[证据链](项目动机与面试证据链.md)。

## 未完成项

- 完整DualPipe、zero-bubble、同GPU多个可执行逻辑stage、完整virtual engine未实现；现有lane pool为在途记账。
- PP没有常驻receive queue，worker串行；独立P2P group组件通过，但矩阵未证明稳定收益。
- 专家loop、count `.tolist()`、多collective仍是热点；native/sourceweight/ownership须分别验收。
- DP缺少成熟超时/恢复和无屏障异步完成队列。
- 多模态serving、量化serving、投机解码未接入，[旧原型复盘](量化与投机解码复盘.md)保留负结果。
- 跨batch shape质量门槛、长输出/真实任务质量及生产网络HTTP SLO仍需补测；本轮官方同机对照已完成但不是严格同构拓扑。
- `per_request` 只验证了受控短输入/1K输入和一个PP2×EP4小样本；它牺牲融合吞吐，不能替代默认批处理的全模型质量与性能验收。

每项性能结论必须对应配置、源码SHA、输入、原始输出、重复轮次和正确性范围。

## 原始材料

- [矩阵JSON](overnight_summary.json)、[长短请求分组](overnight_request_groups.csv)、[跨拓扑分歧](overnight_cross_topology_parity.json)
- [batch-shape 诊断摘要](batch_shape_probe_summary_056.json)与PP2×EP4[SHA清单](batch_shape_probe_056_20261104_203819/local_sha256_manifest.json)
- [CUDA覆盖与汇总](overnight/combined_trace_analysis.json)，各场景原始`trace/cuda_rankN.json`
- [数值诊断](数值正确性排查.md)、[代码结构](当前代码结构审计.md)
- [历史短负载报告](历史报告_旧短负载.md)：旧long16仅196–197输入token，旧DP无逐请求时间，不混用新矩阵。
- `overnight_*.tar.gz.json`保存快照大小/SHA/文件清单；解压后逐文件核验，后续增量保留旧数据。
