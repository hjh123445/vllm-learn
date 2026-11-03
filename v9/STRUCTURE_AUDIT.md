# v9 结构审计（2026-11-03）

范围：Qwen3-Omni-30B-A3B Thinker 文本推理；8×A800 上 EP8、PP2×EP4、DP2×EP4。

## 架构现状

- `engine/core.py`：唯一 scheduler owner；为 worker 广播 step 消息，PP microbatch 按 token budget生成，每步最多两个消息在途。
- `engine/worker.py`：一个 GPU 一个 worker process；初始化 process group、模型组件和 profiler；负责 CUDA/NCCL执行。
- `engine/mq.py`：pickle + 单槽 shared-memory控制消息与 ACK。控制面；模型activation仍由 NCCL管理。
- `engine/frontend.py`：一个DP副本一个独立 supervisor/EngineCore进程树；分配互斥物理GPU。当前离线step_all会join所有副本，存在慢副本barrier。
- `engine/process_groups.py`：所有worker以相同rank顺序建立 TP/CP/EP group；PP仍使用默认P2P process group。
- `layers/linear.py` / `layers/parallel.py`：线性权重装载时按TP shard；RowParallel触发TP all-reduce。
- `layers/moe.py`：本地专家权重形状为 `[E_local, ...]`；EP是变长all-to-allv dispatch/combine，TP×EP路径只在combine之后做TP归约。
- `engine/pipeline_transport.py`：stage间异步P2P、单独comm stream、CUDA event依赖、lane与发送credit；shape header需要worker调用recv后才能收到，payload随后分配，接收端仍同步等待。
- `models/qwen3_moe.py`：文本thinker和global layer stage slice；PP非首stage接activation，最后stage做norm/logits。

## 已修复或已验证

- EP combine恢复top-k route顺序后FP32累加，移除BF16 atomic index_add竞态来源；全模型同输入greedy重复验证通过。
- EngineCore response/broadcast的有限窗口和ACK避免共享槽覆盖/死锁。
- DP副本重复的local Sequence ID会被全局ID映射隔离；worker异常传回调用线程。DP短8、long16、short64重复输出检查通过。
- 全rank一致的new_group调用顺序、显式CUDA设备绑定和group预热通过多卡运行。
- 四卡TP2×EP2合成numerical oracle检查FP32/BF16、不同EP来源token长度及一个空source rank，通过。
- 专家计数在模型显存warmup后重置；profiler window扩展至长prefill真实CUDA activity。
- 初始化CPU thread oversubscription经实测可用OMP/MKL线程数4缓解。

## 仍需优先处理

1. **专家计算效率**：`GroupedExperts` 对非空专家逐个启动PyTorch GEMM，排序、gather、scatter、`.tolist()`及counts/metadata collective带来launch与同步开销。先Nsight归因，再考虑fused grouped GEMM、top-k/router融合和低延迟A2A backend。
2. **EP注意力冗余**：EP ranks持有相同hidden输入并重复计算dense attention/router；这不是成熟attention-DP+EP布局，长prefill会额外浪费算力/显存带宽。
3. **PP调度**：单stage worker串行处理消息，接收仍阻塞；scheduler按整个Sequence分组，超过token budget的请求单独超预算；短请求和microbatch不足有fill/drain bubble。当前不是DualPipe，也没有实测zero-bubble。
4. **PP进程组**：默认group的unbatched P2P会与同组其它操作序列化，NCCL日志明确有warning。后续对比邻接stage专用group或`batch_isend_irecv`，并用数值和故障退出测试验证。
5. **DP前端**：`step_all()`同步join慢副本；缺少启动超时/健康状态/隔离，调度只轮询分配，不按队列长度或请求估时路由。
6. **模型范围**：当前仅加载thinker文本路径；vision/audio权重被loader跳过。无多模态processor、media payload和时空RoPE路径。
7. **质量/功能**：只有重复greedy一致，不代表HF参考logits parity。RTN INT4仍解包到BF16计算；speculative decoding未实现。
8. **可观测性**：CUPTI trace有kernel与stream，但没有SM occupancy/active warp、NVLink bytes、访存带宽等Nsight指标；短profile不支持完整SLO推断。

## 旧的无效接口已处理

`pipeline_microbatches`、`virtual_engine_size`之前只映射到config但不控制Core调度；`--zero-bubble`也没有真正runtime schedule。固定microbatch/zero-bubble仍作为旧API收到非默认值时明确报NotImplementedError；Core实际以`pipeline_microbatch_tokens`动态切分。`build_forward_schedule`只保留为CPU理想时钟测试，不再称runtime schedule。

## 建议验证层次

先CPU topology/API行为测试，再小shape分布式oracle，之后模型同输入同token数输出重复性和HF数值对齐；性能测试用warmup后3次以上结果及P50/P95；最后采集客户端端到端TTFT/TPOT、CUPTI/Nsight轨迹、显存和专家路由长尾。对外汇报每个数字需写清代码hash、GPU、输入长度、输出长度、并行形状、精度及计时边界。
