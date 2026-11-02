# v9 代码结构审查

## 结论

v9 已经具备一个清晰的实验性 EngineCore 方向：主进程持有 scheduler，worker 只执行 forward，控制面使用 shared-memory 消息槽，模型组件在构造时按 TP/EP 切分，PP 使用 stage 间激活传输。这条主线与 vLLM V1 的边界接近。

审查发现，仓库此前同时保留两套执行栈，导致“文档描述的新架构”和“真实 HTTP 入口”不是同一套代码。此次审查已将 `AsyncEngine` 改为驱动 `EngineCore`，修复了 `EngineCore.generate()` 空返回、`model_runner.py` 缺少 `os` 导入、序列 block size 不同步、调度空批次崩溃和 `is_prefill` 状态不更新等问题。

## 当前分层

```text
server.py
  └─ engine.async_engine.AsyncEngine
       └─ engine.core.EngineCore              # 现在的正式执行入口
            ├─ engine.scheduler.Scheduler     # 唯一调度者
            ├─ engine.mq                       # shared-memory 控制面
            └─ engine.worker.worker_main       # 每 GPU 一个 forward worker
                 └─ engine.model_runner.ModelRunner
                      └─ models/qwen3_moe.py
                           └─ layers/parallel.py + layers/moe.py

engine/llm_engine.py + engine/parallel.py + engine/centralized.py
  └─ 历史实验/兼容代码；仍被部分旧测试和 example.py 使用
```

`layers/parallel.py` 是目前应继续发展的并行边界：`ParallelState` 负责拓扑和通信组，`ColumnParallelLinear`/`RowParallelLinear` 负责权重切分，`MoEMLP` 负责 EP 的 token dispatch/combine。模型文件只组装组件，不再由引擎事后替换权重。

## P0：会导致错误结果、崩溃或错误启动的问题

| 状态 | 问题 | 影响 | 处理 |
|---|---|---|---|
| 已修复 | `model_runner.py` 使用 `os.path.basename` 却没有导入 `os` | DeepSeek-V4 路径在初始化时 `NameError` | 已补导入 |
| 已修复 | `EngineCore.generate()` 最后固定返回空列表 | 离线调用看似完成但拿不到结果 | 已按 `LLMEngine.generate()` 返回 text/token_ids/finish_reason |
| 已修复 | `Sequence.block_size` 仍使用类默认值 256 | 配置 512 等 block size 时 scheduler 与 worker 的 block table 不一致 | EngineCore/LLMEngine 初始化时同步配置值；worker 消息携带 block size |
| 已修复 | scheduler 在 KV 暂时不可用时 `assert scheduled_seqs` | 满池或抢占边界直接崩溃 | 返回显式空调度结果，EngineCore 空步安全返回 |
| 已修复 | `Sequence([])` 访问 `token_ids[-1]` | 空 prompt 触发难以定位的 `IndexError` | 改为明确 `ValueError` |
| 已修复 | `postprocess()` 不维护 `Sequence.is_prefill` | chunked prefill 后状态仍像 prefill，PP/调度调试信息错误 | 按 prompt 是否完成逐序列更新 |
| 已修复 | DeepSeek-V4 被当成 Qwen 的多 worker 模型启动 | 每个 worker 可能加载一份完整 DSV4，造成 OOM 或错误并行 | EngineCore 对 DSV4 的多卡组合显式拒绝，并提示使用 `dsv41_a100_runtime` |

## P1：架构和可维护性问题

### 两套执行栈仍然存在

`engine/llm_engine.py`、`engine/async_engine.py` 的旧版本、`engine/parallel.py` 和 `engine/centralized.py` 是早期实验实现。它们继续被 `example.py` 以及若干历史测试导入，造成以下风险：

* 同一个 TP/EP 参数在两个实现中可能有不同语义；
* 测试通过并不代表 HTTP server 走的是同一条路径；
* 面试官阅读时无法判断哪个模块是生产入口。

本次把 `AsyncEngine` 收敛到 `EngineCore`，但旧模块仍应保留为兼容/历史代码，下一步建议：

1. 将 `engine/parallel.py` 和 `engine/centralized.py` 移到 `legacy/`，或在文件头标注“仅历史实验”；
2. 把 `example.py` 改成 EngineCore/AsyncEngine 示例；
3. 把旧测试迁移为 `tests/legacy/`，新测试统一导入 `layers.parallel` 和 `engine.core`。

### DSV4 和 Qwen 的能力边界没有统一接口

Qwen3-MoE 已接入组件级 TP、EP、PP；DSV4 仍是完整模型 forward，使用自己的 loader、HC/MLA/压缩 KV 路径。两者不应共用“看起来都支持任意并行参数”的入口。当前通过 EngineCore 启动校验避免误用，但长期建议为模型定义能力描述，例如：

```python
ModelCapabilities(tp=True, ep=True, pp=True, dcp=True)
ModelCapabilities(tp=False, ep=False, pp=False, dcp=False)  # DSV4 当前路径
```

launcher 根据能力矩阵拒绝组合，并把 DSV4 专用 runtime 作为独立 executor 接口接入，而不是让通用 worker 猜测模型类型。

### DPFrontend 的定位

`DPFrontend` 当前是多个独立 EngineCore 的路由器：每个 DP rank 拥有完整 scheduler、KV cache 和模型副本，请求进入后不迁移。这与 vLLM 的 data-parallel frontend 思路一致，但它不是单个 NCCL world 中的“DP collective”。它适合副本级扩展，不能代替训练式 DDP，也没有跨副本负载迁移。

DeepSeek 风格的前向 zero-bubble 近似、EP=8 和 shared-expert/A2A overlap 的边界与实验方案见 [`DEEPSEEK_INSPIRED.md`](DEEPSEEK_INSPIRED.md)。其中 ZeRO-1 明确列为训练侧概念：推理引擎没有 optimizer state，不能为了概念完整而伪造一个无效的 ZeRO 实现。

### PP/Virtual Engine 的当前边界

PP 已有 stage 间激活传输和微批消息，但 scheduler 仍是单 owner，所有 microbatch 在一个调度步中先发送再收结果；没有 vLLM 那样的每 virtual engine 独立 request queue、抢占状态和真正的 1F1B 调度。因此它是“可验证的微批 transport + lockstep pipeline”，还不是完整生产级 pipeline scheduler。文档和面试表述应保持这个边界。

## P2：后续重构建议

### ModelRunner 职责过重

`ModelRunner` 同时负责模型加载、warmup、KV cache 预算、输入 batch 构造、forward、PP 中间 stage 和采样。vLLM 把这些职责拆到 model loader、executor/worker、input batch、KV cache manager、sampler 等边界中。建议按以下顺序拆分，避免一次性重写：

1. `ModelLoader`：只负责构造模型和权重装载；
2. `KVCacheManager`：负责预算、分配和 attention cache 注入；
3. `InputBatch`：把 Sequence 快照转换成 prefill/decode 输入；
4. `ModelExecutor`：只负责单次 forward；
5. `Sampler`：接收 logits 和 sampling metadata。

### 控制面和数据面

`engine/mq.py` 的 shared-memory 槽只承载 pickle 后的调度元数据，GPU 激活仍由 NCCL/P2P 传输。这是合理的控制面/数据面分离，但当前每个槽只有一个 producer/consumer，不能直接扩展成多 producer queue；扩展前应先加序号、取消和异常关闭协议。

### 失败传播和生命周期

worker 在初始化阶段如果异常退出，EngineCore 仍可能阻塞等待 ready。下一步应给 ready 消息增加超时、worker pid/异常摘要，并在任何 worker 退出时让所有请求收到错误。`AsyncEngine` 已将后台异常转成 stream 的 `error` 和结束哨兵，但 HTTP 层还应把它映射为 5xx。

## 与 vLLM 的结构对照

| v9 | vLLM 对应思路 | 当前差距 |
|---|---|---|
| `EngineCore` | V1 `EngineCore` | v9 是同步 step；vLLM 有更完整的 scheduler output 和事件协议 |
| `worker_main` | `WorkerProc`/executor | v9 每 GPU 一个 Python worker，错误恢复和批处理元数据较少 |
| `layers.parallel.STATE` | `parallel_state` | 已有 canonical rank/group；缺少更完整的 device mesh、上下文和生命周期封装 |
| `layers.moe._ep_dispatch` | Fused MoE dispatcher / DeepEP 路径 | 已避免全卡冗余专家计算，但仍是 Python 编排 + `all_to_all_single`，不是 DeepEP 的 fused dispatcher |
| `PipelineTransport` | pipeline worker / virtual engine | 已有激活传输和微批；缺少完整 1F1B scheduler 与每虚拟引擎状态 |
| `BlockManager` | PagedAttention block manager | 有分页 KV/APC/抢占雏形；缺少更完整的 prefix cache 生命周期和异步 cache events |
| `DPFrontend` | data-parallel frontend | 是独立副本路由，未做跨副本迁移和统一统计 |
| `dsv41_a100_runtime` | 专用高性能 executor | 当前与通用 EngineCore 是两条产品线，应明确隔离 |

## 建议的面试主线

不要把项目包装成“重新实现了 vLLM”。更有说服力的主线是：

> 在 A800 多卡上，如何把 MoE 推理的专家权重、token 通信和 KV cache 生命周期拆成可验证的组件，并把旧的全量加载/冗余计算路径迁移到 EngineCore + 组件级并行。

可验证实验应至少包含：

* TP1/TP2：逐 token 与单卡 greedy 输出一致性；
* EP2：专家权重只在 owner rank，dispatch/combine 前后 token 顺序和输出误差；
* PP2：不同 microbatch 数下 TTFT、TPOT、pipeline bubble；
* DP2：两个独立 EngineCore 的吞吐和请求路由；
* DSV4：专用 runtime 与通用 EngineCore 的能力边界、显存占用和失败提示。

每项实验应同时记录配置、GPU 型号、dtype、prompt token 数、生成 token 数、TTFT、TPOT、吞吐、峰值显存和输出校验方式。没有 A800/NCCL 实测时，只能声称“协议和数学单测通过”，不能声称多卡性能已经验证。

## 本次验证

当前环境没有 PyTorch、Transformers、pytest 和 CUDA，因此本次只能完成 Python AST/`compileall`、纯 Python 调度协议和 shared-memory 协议检查。TP/EP/PP、DeepEP 效果以及 DSV4 数值正确性必须在 A800/NCCL 环境复测。
