# Profiling、Perfetto 流水线阅读与 Triton 快路径审计

当前CLI与最新验证见 `BENCH_PROFILE_CLI.md`：已补齐bench offline/serve/trace-check、server --profile；CUDA13缓冲区flush兼容处理后，TP2和PP2TP2真实CUDA/NCCL采集通过。本文历史trace说明保留其原始范围。

EP拓扑更新说明：当前已改为EP=DP×TP，旧trace保留历史含义；新拓扑接口见EP_DP_TP_IMPLEMENTATION.md。

2026-11-05。本次核对公开源码、现有原始数据并修复离线工具，没有启动新的 GPU 实验。官方行为以 **vLLM v0.31.0** 为阅读版本；历史性能基线使用 v0.14.1，不能混用版本结论。

## 1. 先回答：之前哪里不科学

问题不是用了 `record_function`，而是把不同层次的证据混在一起：

| 证据 | 能说明什么 | 不能据此推断什么 |
|---|---|---|
| `pipeline_rank*.json` | Python forward/P2P 调用的 CPU 区间 | GPU kernel 时长、CUDA stream 或计算通信重叠 |
| `cuda_rank*.json` | 成功采集时，含 CPU ATen、CUDA API、GPU kernel、内存拷贝与通信 kernel | 文件名有 CUDA 就保证采集成功；某些历史 rank 实际只有 CPU 事件 |
| GPU-only / compact 派生文件 | 减少体积、突出某些轨道 | 被主动过滤掉的 ATen 不代表程序没有调用 ATen |
| 无 profiler 的吞吐结果 | 给定输入、设备与配置下的吞吐/延迟 | 自动解释性能原因 |

原来的 `tests/merge_pipeline_trace.py` 同时读取手工 `perf_counter` 文件和 Kineto 文件，却没有校验时钟；直接拼接各 rank 的 pid/tid 和 flow ID，还把 CPU forward/P2P 区间之和命名成计算/通信时间。这些都已修正。

修正后的工具：

- 默认只读 `cuda_rank*`；`--kind pipeline` 单独处理手工 CPU trace。
- 每个来源独立映射 pid/tid、flow/correlation ID，保留同来源内部关联。
- 使用 Kineto `baseTimeNanoseconds` 校正时间基准，然后只减去**一个全局共同起点**；保留所有持续时间。
- 多来源要求显式 `--same-host`，含义是用户确认同一次运行、同一主机。跨主机时钟需要外部校准；本工具不自动处理。
- 混合时钟拒绝合并。缺少基准的多个文件，必须先确认共同微秒时钟才可使用 `--shared-timestamps`。
- 保留 ATen、CUDA runtime、kernel、memcpy、GPU annotation 与 flow；不把派生视图冒充完整原始 trace。
- GPU 统计采用区间并集/交集，CPU 标注不计入 GPU 时长；空白占比不命名成 PP bubble ratio。

## 2. 官方 vLLM 如何 profile

公开源码入口：

- [官方 profiling 文档](https://github.com/vllm-project/vllm/blob/v0.31.0/docs/contributing/profiling.md)
- [WorkerProfiler / TorchProfilerWrapper / CudaProfilerWrapper](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/profiler/wrapper.py)
- [ProfilerConfig](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/profiler.py)
- [GPUWorker：annotate_profile 与 execute_model](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu_worker.py)
- [layerwise_profile 专用诊断工具](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/profiler/layerwise_profile.py)

整体链路是：服务/离线入口启动 profiler → 各 worker 自己采集 → 执行模型 → 各 worker 停止、导出。父进程上的 profiler 不会自动采到子进程 CUDA 工作。

`TorchProfilerWrapper` 在 NVIDIA GPU 上默认采 CPU+CUDA，支持 wait/warmup/active、延迟开始和最大迭代数。停止时导出。`GPUWorker.annotate_profile()` 会根据调度输入添加 context/generation 请求数、token 等执行批次信息；Torch 后端使用 `record_function`，CUDA profiler 后端使用 NVTX range。

这说明：**官方也使用显式范围标注，但无需为了启用 profiler 给每个模型重写 forward。** ATen、CUDA API 和 GPU 活动由底层 profiler 自动采集；模型层号、某段路由算法的语义则需要模块信息、调用栈或少量语义标注。

专用 `layerwise_profile` 启用 `record_shapes/with_stack/with_modules`，构造模块树和 CUDA 关联，用于诊断。不要据此假定所有 eager、编译模型和 CUDA Graph replay 都会自动产生完整可读模块树；需要检查实际 trace。该工具依赖 PyTorch 内部 profiler 类型，也应固定版本。

### 官方服务采集示例

下面是参考命令，**本轮没有执行**。实际安装版本先检查 `vllm --version` 与 CLI 帮助。模型路径替换为实际公开权重路径。

```bash
vllm serve "$MODEL" --tensor-parallel-size 8 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./trace_vllm","torch_profiler_record_shapes":false,"torch_profiler_with_stack":false}'

# 等初始化完成，先用相同负载预热，再开始采集。
curl -X POST http://127.0.0.1:8000/start_profile
# 在这里运行固定输入、固定到达模式的请求客户端。
curl -X POST http://127.0.0.1:8000/stop_profile
```

`vllm bench serve --profile` 也可控制采集。默认不要在吞吐测试中开启 profiler；shape、stack、内存追踪都会增加开销。更不能同时跑两组基线抢同一组卡。

Nsight Systems 的对应方式：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile \
  --trace=cuda,nvtx,osrt --trace-fork-before-exec=true \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=repeat -o qwen_systems \
  vllm serve "$MODEL" --tensor-parallel-size 8 \
  --profiler-config.profiler cuda
```

通过相同 start/stop 控制采集。`.nsys-rep` 用 Nsight Systems 打开；Chrome JSON 用 Perfetto。具体 NCCL 插件、网络指标能力取决于 Nsight/NCCL 版本，不能仅凭 `--trace=cuda` 宣称已采到 InfiniBand/NVLink 带宽。

Nsight Systems 用于 CPU 调度、CUDA stream、NVTX 和通信等待的系统级分析；Nsight Compute 用于已定位 kernel 的带宽、指令、占用率等。后者可能串行化/重放 kernel，不用于证明原程序的并发和端到端吞吐。

## 3. 我们应该如何组织标注，而不用到处改模型

建议把两件事分开：**自动记录运行事件**，**标记业务/算法语义**。

1. 每个worker进程内的独立观测Runner管理profiler生命周期，engine和模型文件都不负责创建、停止和导出profiler。
2. 在模型外遍历 `named_modules()`，按模块路径给 decoder block、Attention、MoE、router 注册可移除的 pre/post hook。在 profiling 模式下打开命名范围，结束时移除 hook；关闭 profiling 时不安装 hook。
3. 只给目标边界装 hook，避免每个小 Linear 都加一层冗余嵌套。名字来自实际模块路径，例如 `model.layers.24.self_attn`，再附加 rank/stage；不硬编码“48层”作为通用逻辑。
4. Router本身是module，可以单独标注；EP dispatch、本地专家compute_sorted和combine仍是内部方法，module hook不会自动拆开。本次按要求删掉模型/算子内部的显式范围，不再承诺这些内部阶段有独立命名；实际ATen、CUDA/NCCL kernel仍由底层采集，不能把整个MoE范围当作专家GEMM时间。
5. 批次范围带 `batch_id/request_count/query_tokens/context_lengths/prefill/decode`。当前 CPU trace 中的 lane 是兼容字段，不是 CUDA stream，也不是 v0 virtual engine。
6. 通过 CPU ATen/runtime 的 External id、correlation/flow 找到真正 GPU kernel，再把它归属到发起操作所在的 layer/Attention/MoE。GPU kernel 可晚于 CPU range 结束；不能只按 CPU/GPU 时间包围关系归属。

hook 只加范围，不在每层 `.cpu()`、`.item()`、复制激活或 `cuda.synchronize()`。精度取证的 activation capture 是另一种高开销模式，不与性能采集合并。编译模型和 CUDA Graph 可能融合、消除或改变模块调用，需要另做 replay/capture 对照；不能为了画图关闭 graph 后，把 eager 性能当生产性能。

统一模块hook已实现于 `utils/module_profiling.py`。`utils/observed_runner.py` 中的独立观测Runner在profiling生命周期内安装/移除，单卡 `tests/profile_trace.py` 也使用同一模块采集器。模型、MoE和ParallelState中的record_function、手工逐层计时、专家统计字段与导出方法均已删除；ModelRunner的重复范围也已删除。正常推理不安装这些profiling hook。

新CUDA trace的语义范围示例为 `stage1.layer24`、`stage1.layer24.attention`、`stage1.layer24.moe`、`stage1.layer24.moe.router`。hook不复制张量或同步CUDA，forward异常也会闭合范围。仅支持实际执行nn.Module调用的eager路径，不保证compile或CUDA Graph replay中的逐层可见性。

`pipeline_rank*.json`现在由独立观测Runner/Transport生成，记录forward和P2P调用的手工CPU时间线，已移除模型提供的逐层CPU事件。新版本异步send范围只到send_forward调用返回，不再把排队到完成等待的整个区间标为send；它与旧范围定义不同，均不代表实际GPU传输时间。逐层信息应查看启用模块hook的 `cuda_rank*.json`。旧trace和旧实验数据保持历史原样。本次没有重新采集GPU trace。

旧 `VLLM_LEARN_PROFILE_EXPERTS/PROFILE_DIR`、`--expert-counters` 路径已移除；不再生成 `qwen_expert_profile_rank*.json`。避免每层统计时 `.cpu()` 干扰调度。历史专家分布报告只对应其原始源码与数据，不能认为新采集仍包含这些计数。EP oracle测试在测试边界统计实际计算route数，不再依赖生产模型的统计字段。

### engine也不依赖采集器

`engine/worker.py`、`model_runner.py`、`pipeline_transport.py`中的profiler生命周期、采集环境变量、手工trace列表、显存导出及logits/activation诊断全部移至 `utils/observed_runner.py`。未使用的 `engine/pipeline_schedule.py` 中TraceEvent/PipelineTrace类已删除。正常EngineCore即使继承了PROFILE/TRACE/LOGITS等环境变量，也不会自行采集。

EngineCore仅提供通用 `runner_cls`、`transport_cls` 替换入口及runner shutdown生命周期，不导入或判断任何采集器。默认仍是ModelRunner和PipelineTransport。实验脚本显式选择可导入的顶层类，spawn后各进程各自观测；没有全局替换模型forward、engine类或torch通信函数。

```python
from engine.core import EngineCore
from utils.observed_runner import ObservedModelRunner, ObservedPipelineTransport

# 仅实验入口选择；相关采集目录/窗口环境变量也只由独立观测工具读取。
core = EngineCore(model, tp_size=4, pp_size=2,
                  enable_expert_parallel=True,
                  runner_cls=ObservedModelRunner,
                  transport_cls=ObservedPipelineTransport)
```

`capture_parallel_window.py`、bench脚本、teacher-forcing和reference-generation诊断脚本已显式选择这些实现。HTTP AsyncEngine默认仍走普通Runner，没有隐式启用profile。单卡外部 `torch.profiler.profile` 仍能采到底层ATen/CUDA；模块标注需由采集脚本安装ModuleProfileRanges。

logits诊断改为观测Runner在sampler前注册pre-hook，利用当前请求信息保存终止query行；activation沿用外部模块hook。warmup结束后才安装，独立DP目录仍按副本编号分开，shutdown移除hook。正常路径不做这些CPU张量拷贝。

请求完成时间、服务TTFT/ITL时间戳、超时检测、KV显存预算、P2P CUDA event/stream依赖和NCCL销毁前的必要同步属于运行逻辑，仍在engine内。代码职责隔离不等于profiling零开销；采集额外引入的同步、hook和导出仅在显式选用的观测实现中发生，性能与诊断实验仍须分开。

## 4. Perfetto 是一个时间轴、多条并行轨道

横轴只有一条共同时间标尺，纵向可以有很多行：CPU 进程/线程、GPU/rank、每个 CUDA stream。**同一横坐标的不同轨道有事件，就是同时发生。**

下面是示意，字符不是实测数据：

```text
共同时间 →       t0         t1         t2         t3
GPU0 compute     [MB0 S0]   [MB1 S0]   [MB2 S0]
GPU1 compute                [MB0 S1]   [MB1 S1]   [MB2 S1]
GPU0 comm                   [发送MB0]
```

这里不同 GPU 上的 stage 工作重叠，就是 PP 并发。每张 GPU 同时只跑一个计算 kernel 也可以存在跨卡 PP 并发。

同卡计算/通信重叠则是：

```text
共同时间 →       0          10         20         30 us
GPU0 stream7     [       GEMM       ]
GPU0 stream23          [      NCCL      ]
                       ^ 时间交集 ^
```

同一 CUDA stream 的普通 kernel 按顺序执行；不同 stream 只是具备并发条件，数据依赖、event wait、SM/寄存器等资源仍会阻止重叠。一个 GPU 进程可以管理多条 stream，并非一个进程只能串行。

### 为什么 merge 后像串行

- 原始事件本来就串行：所有工作在一条 stream，或者明确等待依赖。
- 只打开了 `pipeline_rank`，或只看 CPU 主线程/折叠后的进程行。
- 多 rank 共用 `pid=0,tid=7`，被错误放到同一个轨道。
- 将每个 rank 分别归零，或按 rank 累加偏移，把真实先后关系改写了。
- 简化视图主动去掉了 stream、CPU ATen 或 flow。
- 采集窗口太短，或某个 rank CUDA 采集失败。缺失不能解释成该卡空闲。

### 打开本次修正文件

打开 `https://ui.perfetto.dev/`，选择 Open trace file，加载：

`outputs/verified_parallel/profile_audit_20261105/pp2ep4_two_stages_full_cpu_cuda.json.gz`

查找并展开 `rank0 / stage0 / ... GPU device 0` 和 `rank4 / stage1 / ... GPU device 4`；两组都应有 stream 7、22、23。把两个 GPU 轨道都展开，缩放到毫秒级，比较同一横坐标。CPU 主线程另看 layer/Attention/MoE 标注，点击 kernel 的 flow 回查启动操作。不要把 GPU annotation 包围范围的时长，直接当成里面有效 kernel 的时间总和。

本文件约20 MB压缩，来自旧版 **2026-11-04 rollout** 的两份原始 worker trace，只是重新正确合并，**不是新的 V1 PP 实现性能实验，也没有覆盖全部8个rank、HTTP客户端或完整服务生命周期**。

### 本次真实解析结果

使用官方 Perfetto trace processor v59.1 读取，没有解析错误：

| 类型 | 数量 |
|---|---:|
| CPU op | 498,482 |
| CUDA runtime | 108,329 |
| GPU kernel | 64,331 |
| GPU memcpy | 7,621 |
| GPU memset | 30 |
| CPU user annotation | 4,774 |
| GPU user annotation | 3,221 |
| 解析出的 flow | 71,982 |

共877,953条Chrome事件，包含metadata与flow事件。不是只有 `record_function`。解析结果见 `perfetto_validation.json`，来源SHA、时钟变换和区间统计见 `merge_audit.json`。

两个 rank 的非 NCCL kernel 在共同时间轴上的交集约 **11.77 ms**，说明有跨卡 kernel 并发；同卡 NCCL 与非 NCCL kernel 交集都为 **0**。前者不是完整 stage 利用率，后者也只覆盖本次捕获窗口。这两者完全可以同时成立。

## 5. 重叠、气泡与 Attention/MoE 占比如何量化

### 同卡通信重叠

在指定 GPU 和完整观测窗口 W 内，先对每类实际 GPU kernel 的时间区间求并集：

- C：非 NCCL kernel 的时间并集；其中包含路由/重排，不全是 GEMM。
- N：NCCL kernel 的时间并集；其中可能包含等待其他 rank。
- 重叠时间：`|C ∩ N|`。
- 通信时间被覆盖比例：`|C ∩ N| / |N|`。没有 NCCL 则为未定义。

CPU `isend()` 很快返回、CPU 的两个范围重叠、两张卡同时有事做，都不等于同卡 C/N 重叠。即使交集大，也需要无 profiler 的 A/B 验证：并发可能抢 SM/带宽，导致 kernel 变长，最终性能反而变差。

历史截图对应 EP8 rank0、layer12 MoE 的50个kernel全在stream7：45个非NCCL、5个NCCL，交集为0。非NCCL并集约191.94us、NCCL并集约1408.32us；CPU MoE范围约3762.35us。三者回答不同问题，不能互相替代。

### PP bubble

先确定相同 batch/微批跨 stage 的依赖链：S0执行MB0 → activation send → S1 recv完成 → S1执行MB0；同时观察S0是否能开始MB1。用 batch ID 关联，不能仅凭不同阶段的“第几个循环”编号相同就认为是同一批。

对每个 GPU，在共同且完整的窗口内求所有 kernel+copy 的活动并集 A，`1-|A|/|W|` 只是观测到的 GPU 无活动比例。它不能自动命名为 PP 气泡，因为白区也可能是 CPU starvation、kernel launch 间隔、CPU回读、输入准备、请求不足、采集缺失或别的进程干扰。

还需要这些证据来归因：

| 白区/等待特征 | 下一步核对 |
|---|---|
| 开头逐stage启动、结束逐stage清空 | fill/drain；区分稳态与整个请求周期 |
| 下游反复等相同stage的激活 | stage计算量/通信量不均；比较相同MB的层耗时 |
| 所有stage一起出现白区 | scheduler、CPU dispatch、token反馈或数据准备 |
| NCCL很长而上游kernel未完成 | straggler与依赖，不能直接断言网络带宽不足 |
| CPU DtoH API长、实际copy仅几个字节 | 等前置GPU工作；减少同步/控制回读，不能按API时长计算链路带宽 |
| 未达到PP深度的可运行独立批次 | 请求依赖与在途批次数；增加batch queue深度不保证单请求加速 |

理想等长 forward-only pipeline 的简单模型：P个stage、M个独立微批、每段用时相等且忽略通信时，平均空闲比例为 `(P-1)/(M+P-1)`。动态混合长度、decode反馈、EP负载和CPU开销都会破坏这些假设，不能把这个公式当实测值。新V1队列的效果应重新采集，不能套旧trace。

### Attention/MoE占比

按 layer/phase/batch/context length 分桶，用CPU启动关联给GPUkernel归属；同一个kernel只归一次。嵌套的layer→MoE→expert范围不可重复累加。输出Attention、router、expert compute、EP dispatch/combine/NCCL和未归属部分。

多stream并发时，kernel时长之和可能超过端到端时间：同时报告总工作时长、时间并集和关键路径等待，不强迫所有占比相加等于100%。跨rank既看平均也看最慢rank和专家token分布。

短decode的有效M很小，MoE每个专家实际收到的M还更小；可能由GEMM进入更接近GEMV的效率区间。长context的decode还受KV读取影响。仅看到batch小不能证明带宽瓶颈，需要结合shape、kernel、显存流量和硬件指标。

## 6. Triton 为什么没有默认开，具体有什么问题

### 当前真实开关

`layers/kernels.py` 的 `REFERENCE_MATH` 默认1，RMS/QK分别继承；`layers/attention.py` 的Attention backend默认 `sdpa_reference`。这些开关在模块导入/对象创建时读取，实验时使用独立进程，不能导入后改env期待已创建对象改变。

这不是A800不支持Triton，也不是所有Triton入口被全局禁止。`silu_and_mul` 是独立入口，但当前MoE主要路径直接使用PyTorch激活，不能仅凭文件里定义了它，就说实际运行调用过它。

### 整模证据：失败的快路径不能当默认优化

本轮重新从原始JSON计算，与**匹配cached执行路径的HF terminal-head raw argmax**逐步比较：

| 数学/Attention路径 | 匹配下一token位置 / 48 |
|---|---:|
| 参考RMS+QK / SDPA | 48/48 |
| 融合RMS+QK / SDPA | 46/48 |
| 融合RMS+QK / Triton Attention | 42/48 |
| 仅QK参考 / SDPA | 44/48 |
| 仅RMS参考 / SDPA | 46/48 |
| 参考RMS+QK / Triton Attention | 40/48 |

来源为 `math_ablation_056_20261104_140005` 与 `followup_056_20261104_133143`；逐文件SHA、参考文件与重算结果见 `kernel_oracle_audit.json`。

必须注意：这是12条合成prompt×4个teacher-forced位置的**历史**对照，不是任务准确率，不是本轮重跑。`requests.json` 自带的 `expected_next_token` 来自另一参考路径，直接对该字段计数会得到不同结果；本表明确使用HF terminal-head的 `raw_argmax_ids`，按(workload,index,prefix_tokens)连接。也不能因为单开某个参考算子更差，就认定它有新的逻辑bug；中间舍入改变后，路由/top-k的误差传播不是线性可加的。

这些实验支持“快路径尚未达到当前HF对齐门槛”，**不证明所有Triton代码逻辑错误，也未把差异定位到唯一一行**。已另行修复的RoPE建表设备、router GEMM形状问题，和跨batch BF16 GEMM差异也不能混为Triton故障。

### 源码与测试缺口

1. RMS/QK融合改变归约顺序、rsqrt与中间舍入。已有显式dtype转换和 `enable_fp_fusion=False`，但这不保证与HF逐位相同。QK的两半平方分别归约再相加，与全head归约顺序也不同。
2. Triton Attention是独立在线softmax实现。prefill Q/K保持模型dtype，累加与softmax部分使用FP32，P在PV乘法前转成V dtype；不能用“FP32累加”推出等同SDPA，也不能说所有乘法都变成FP32。选择后端和矩阵形状也会改变数值。
3. kernel入口对shape/stride的通用校验不足。QK的 `tl.arange(0,HQ/HK)` 要求受支持的头数形状；head_dim、GQA整除、连续布局、cache/position索引约束不能靠任意调用者猜测。当前Qwen常用形状符合部分约束，不等于任意TP/模型都兼容。
4. decode按(请求,Q head)一个program串行扫描KV；小batch、长context时有并行度不足的风险。prefill固定BM/BK=32、4 warps，没有覆盖不同负载的充分调参证据。这些是优化候选，尚未测出收益。
5. `tests/test_kernel_reference.py` 是导入即执行CUDA的脚本。默认env下RMS/QK走参考分支，**不等于测过Triton分支**；还导入却未调用SiLU。测试形状有限，BF16使用较宽容差，缺少系统的TP头数、边界和整模门槛。
6. 旧docstring声称“全部Triton”“FP32逐位一致”，和当前默认/证据不符，本轮已修正文案，未偷偷开启快路径。

### 重新启用的验收顺序

固定同一输入/权重/shape，先单算子对照，再替换一个算子跑逐层首次偏差，最后做cached teacher forcing和自由生成；同时覆盖prefill/decode、分页乱序、prefix/chunk、长短输入、TP形状和空负载。记录误差分布、router top-k差异、logit margin与首次分叉。位级一致与语义质量门槛分别定义，不能只看 `assert_close`。

参考基线→只开RMS→只开QK→只开Attention→组合的因子消融中，每组只改一个开关。对通过门槛的形状再独立跑无profiler A/B与短trace，验证launch数、内存读写、kernel时长及端到端toks/s；不支持的形状显式fallback。未通过前保持参考默认。

## 7. 复现工具与采集边界

在仓库根目录合并同次单机采集的原始worker文件：

```bash
python v9/tests/merge_pipeline_trace.py \
  --input-dir outputs/RUN/trace --same-host \
  --output outputs/RUN/full_cpu_cuda.json.gz

# CPU手工轨迹单独导出，不混进CUDA文件：
python v9/tests/merge_pipeline_trace.py \
  --input-dir outputs/RUN/trace --kind pipeline --same-host \
  --output outputs/RUN/cpu_submission.json

python -m pytest v9/tests/test_trace_merge.py -q
```

本次8项离线测试通过：跨rank同时事件仍同时、不同base正确对齐、关联ID不串rank、区间交集不重复计数、混合时钟拒绝、未知时钟需核实、CPU-only不冒充GPU、空输入拒绝。

`tests/capture_parallel_window.py` 是当前分布式采集入口；它在worker内开启CPU+CUDA profiler，保存inputs/config/outputs，和吞吐测量分开。例如下面是**待运行**的新采集命令，不是本次文件的原始命令：

```bash
VLLM_LEARN_PROFILE_RECORD_SHAPES=0 \
python v9/tests/capture_parallel_window.py \
  --model "$MODEL" --inputs "$INPUTS" --output outputs/NEW_RUN \
  --pp 2 --tp 4 --enable-expert-parallel --workload short64 --requests 32 \
  --start-message 3 --messages 16 --max-kv-blocks 128
```

该脚本当前强制每请求最多4个输出token；一次32请求不保证所有负载都能形成足够长decode稳态。message窗口也是每worker本地计数，开始边界会错开；需要核查各stage共同完整batch覆盖，不能直接当端到端连续窗口。现有worker采集边界有同步与导出开销，不能拿profile run作吞吐基准。

未来科学对照至少包括：同机同输入、独占测量时段、相同生成策略；B=1/8/32/128、短/长/混合context，prefill和decode分开分桶，至少多次无profile测量；再对最有解释价值的窗口采少量trace。保存代码/权重/输入指纹、环境、原始失败、准确性门槛与每个rank的CUDA采集完整性。

本轮产物是离线工具修复、历史trace重新对齐、真实Perfetto解析与kernel门槛复核，**没有新的吞吐提升数据**。
