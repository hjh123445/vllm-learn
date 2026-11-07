# Profile：正常运行时加一个 --profile

更新：2026-11-07。命令在仓库根目录运行。采集需要 Linux/CUDA；分析、合并只需 Python 标准库。

CUDA Graph 补齐的数值排查和严格验证见 [CUDA_Graph补齐精度排查.md](CUDA_Graph补齐精度排查.md)。离线 benchmark 支持 `--decode-graph --batch-invariant --profile`。Graph replay 有真实 CPU launch、GPU stream/kernel 和 NCCL 记录，但不会再次执行逐层 Python hooks；审计对含 replay 的 trace 将 `layer_attribution_verified` 标为 false，不能把 replay 的 GPU 活动完整性说成所有图节点均已逐层归属。严格模式的固定 GEMM、固定归约会改变数值策略与性能，A/B 两侧必须保持相同策略。

## 1. 正常 serve 顺便采集

```bash
export MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 终端1：普通服务参数照常设置；额外加 --profile 即开始记录。
python v9/server.py --model "$MODEL" --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 --enable-expert-parallel \
  --max-batch-size 128 --max-num-batched-tokens 512 \
  --pipeline-microbatch-tokens 128 --max-concurrent-batches 2 \
  --port 8000 --profile --profile-dir outputs/http_pp2ep4

# 终端2：正常发请求；请求总数/并发度与 GPU 单批大小是不同参数。
python v9/bench.py serve --base-url http://127.0.0.1:8000/v1 \
  --output outputs/http_pp2ep4_load --requests 256 --concurrency 32 --max-tokens 32
```

客户端完成后，在服务端按 Ctrl+C（或向该服务发送 SIGTERM），等待“Profile coverage verified”和正常退出，再分析文件。**导出发生在关闭时**，SIGKILL/断电无法保证导出。服务持续运行期间，后续请求也会继续记录。

可以只加 `--profile`，省略 `--profile-dir`：自动创建 `outputs/server_profile_时间_随机后缀/`，启动时打印路径。显式目录中的 `trace/` 必须为空，防止混入旧 rank。`--max-batch-size` 是 `--max-num-seqs` 的别名。HTTP 服务目前不提供 DP 参数；DP 使用下述离线入口。

默认 `--profile-start-message 1 --profile-messages 0`，其中 **0 表示采到关闭**。窗口按每个 worker 的模型调用计数，不是请求数、token 数或层数。只有主动限制文件大小时才设 `--profile-messages 12` 等有限窗口；这时窗口外的执行不在 trace 中。

`--profile` 不替换调度器、不修改请求参数、不改变数值后端；初始化和模型预热不采。采集本身有开销，不能保证和无 profile 时的实际 batch 组合/时间相同。全程采集随运行时长增加内存与文件体积。

## 2. 正常 offline bench 顺便采集

```bash
python v9/bench.py offline --model "$MODEL" --output outputs/offline_pp2ep4 \
  --tp 4 --pp 2 --enable-expert-parallel --max-batch-size 128 \
  --workloads short128,mixed64,context4k8 --repeats 3 \
  --budget 512 --microbatch-tokens 128 --max-concurrent-batches 2 --profile
```

三个负载、各三轮都照常执行，全程记录在同一次各 rank 的 trace 中；不会强制改成一个负载或一轮。`inputs.json` 保存实际 token IDs/输出长度/到达 step，`results.json` 保存逐请求生成结果与诊断时间。可用 `--inputs 路径` 重放输入。离线性能模式额外运行请求预热；profile 模式直接采所选负载，避免有限窗口被额外预热耗尽，模型初始化预热始终排除。

**吞吐、TTFT/TPOT/P95/P99 必须另跑无 `--profile` 的同负载实验。** profile 下这些汇总指标为 null、`benchmark_valid=false`，HTTP 客户端也会通过 `/health` 自动识别服务是否开启 profile。不把受采集干扰的时间当性能成绩。

## 3. 检查，然后合并进 Perfetto

```bash
python v9/tools/profile_analyze.py --input-dir outputs/http_pp2ep4/trace \
  --output outputs/http_pp2ep4/analysis --expected-ranks 8 --layers 48 --require-nccl --same-host
python v9/tools/profile_merge.py --input-dir outputs/http_pp2ep4/trace \
  --same-host --output outputs/http_pp2ep4/trace.json.gz
```

把 `trace.json.gz` 拖入 [Perfetto](https://ui.perfetto.dev/)。展开各 `rankN / stageS / CPU pid` 和 `GPU device`，pin 住不同卡的 CUDA stream。**多卡是多行共用时间轴，可以并列展示**；合并不会首尾拼接或分别归零。工具按同机 Kineto `baseTimeNanoseconds` 对齐，对 pid/tid、correlation/flow 加 rank 命名空间，保留 CPU→GPU 关联；跨机器时钟不在此自动校准。

| 文件 | 用途 |
|---|---|
| `trace/cuda_rankN.json` | 每 worker 的真实 CPU 线程、ATen、CUDA API、GPU stream/kernel、memcpy 和 NCCL kernel |
| `trace/capture_rankN.json` | 独立记录实际执行次数、采集消息清单、成功状态、stage/层区间、CUPTI flush 状态 |
| `trace/audit_rankN.json` | 每消息 CPU/GPU Attention+MoE 覆盖、launch 关联与缺口 |
| `trace/trace_validation.json` | 正常关闭时的全 rank 验收；失败会使命令报错 |
| `trace/pipeline_rankN.json` | **仅主机 run/send/recv 调用范围**，不含 ATen/CUDA，不能用来看 GPU 重叠 |
| `analysis/summary.json` | 逐层/逐消息 GPU 时长、kernel 热点、CUDA API、stream、空隙和计算通信交集 |
| `analysis/report.md` / `timeline.html` | 中文检查报告 / 多卡 GPU 共同时间轴概览 |
| `trace.json.gz` | 无删减合并所有原始已记录事件，供 Perfetto 查看 CPU/stream/flow 细节 |

profile 导出先写 `.partial`，成功后原子改名；不要打开仍在写的文件。`timeline.html` 是像素分桶概览，精确重叠以完整 trace 为准。

## 4. 采集边界与实现隔离

普通 `server.py` / `bench.py offline` 只在启用时选择 `utils/observed_runner.py` 的采集适配器。`engine/`、`models/`、`layers/` 不含 profiler/record_function 或诊断环境依赖。`utils/module_profiling.py` 临时安装 module hooks：

```text
rank0.stage0.message1.mb0.prefill       一次 worker forward
  stage0.layer0                       全局 decoder 层号
    stage0.layer0.attention
    stage0.layer0.moe
      stage0.layer0.moe.router
```

PP2 对48层模型：rank0–3执行层0–23；rank4–7执行层24–47。本卡没执行的层不会生成标签。hooks 是 CPU 范围；ATen/CUDA/NCCL 由 PyTorch profiler/CUPTI 采集。GPU 归层沿 CPU 范围→launch correlation/External id→GPU kernel，**不按横轴恰好落在 CPU 范围下面猜归属**。

每个 GPU worker 各自采集其 CPU 线程和所有已记录的 CUDA stream。当前不自动包含父调度进程、HTTP/tokenizer 主线程、初始化/权重加载、媒体 encoder；因此这是 worker 执行轨迹，并非整个 HTTP 服务全生命周期。前端/媒体等全进程系统分析需另用 [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

仅在窗口起止同步 GPU，关闭时刷新 CUPTI；不在每消息/每层插入同步，避免人为破坏原流水。CUDA13/Torch2.9.1 的该远端环境曾出现最小矩阵乘法也只导出 CPU 的问题：GPU 完成不等于 CUPTI 活动缓冲已交给 Kineto。独立适配器在 stop 前执行设备同步和 `cuptiActivityFlushAll` 兼容处理，保存状态码并核对真实 kernel；目前证据定位到该环境的缓冲交付链路，不能断言所有版本都有同一根因。

## 5. 如何确认没漏卡、漏消息、漏尾层

关闭后自动验收，也可重跑：

```bash
python v9/utils/trace_check.py --input-dir outputs/http_pp2ep4/trace --expected-ranks 8 --require-cuda --require-nccl
```

先核对精确 rank 集合与重复文件；再根据独立执行清单核对窗口内每条消息，而不是仅检查“文件里有一些 CUDA”：CPU 消息整条缺失、GPU 消息整条缺失、任意层 Attention/MoE CPU 或 GPU 归属缺失、kernel launch 没有关联 GPU 活动、forward 异常、未正常关闭均失败。空 DP worker 真实只执行 MoE，不凭空要求 Attention。

`cuda_verified` 仅说明有 kernel；`completeness_verified` 才表示上述逐消息覆盖检查通过。**通过也不是数学上证明任意事件零丢失**，不代表数值/性能合格。图执行等未建立归层证据的路径不能冒充验收通过。

旧30B trace 的末次 decode 缺 GPU 尾层/整条消息，现已能明确检出，不能靠合并补出丢失活动；旧 capture 缺独立层区间/执行清单也不能认证。新八卡自生成小模型已用于首 prefill 到末 decode、正常退出及生成结果不变验证，证据见 [输出索引](../outputs/README.md)；不把小模型功能验证当30B速度或质量成绩。

## 6. 在轨迹中判断流水线与重叠

- **同卡计算通信重叠**：不同 stream 的 GEMM 等计算 kernel 与 NCCL **GPU 执行区间在横轴相交**。CPU 异步 API 提前返回或两个 CPU 范围交叠不算证据。报告先求区间并集，计算 `|compute ∩ NCCL| / |NCCL|`；NCCL 可能等待 peer，交集不等于有效通信完全隐藏。
- **跨 stage 流水线**：GPU0 执行 stage0 的批B，同时 GPU4 执行 stage1 的批A；与同卡通信重叠是两件事。同一 CUDA stream 按序执行，同一进程不同 stream 可以并发，取决于依赖和硬件资源。
- **气泡**：先确认双方都在采样的共同稳态窗口、队列有可执行请求，再追上游 activation、recv 等待、CPU launch/IPC 同步、MoE 长尾。相同输入对照 `--max-concurrent-batches 1` 与 PP size，分别测无 profile 吞吐并各采一份 trace。空白也可能是主机供给不足、采样边界、重算或 NCCL 等 peer，不能把 `observed_inactive_fraction` 直接叫 PP bubble 或 SM idle。

## 7. 拓扑与可选工具

总 worker 数为 `DP×PP×TP`，启用 EP 时每 stage 专家组大小为 `DP×TP`，EP 不是额外设备轴。

| 拓扑 | offline 参数 | 看什么 |
|---|---|---|
| TP8 | `--tp 8` | 层内 TP collective |
| TP8+EP8 | `--tp 8 --enable-expert-parallel` | 单 stage，专家 dispatch/combine |
| PP2×TP4+EP4 | `--pp 2 --tp 4 --enable-expert-parallel` | 两 stage，多卡 P2P 和专家通信 |
| DP2×TP4+EP8 | `--dp 2 --tp 4 --enable-expert-parallel` | 两 EngineCore，EP 跨 DP×TP |
| DP2×PP2×TP2+EP4 | `--dp 2 --pp 2 --tp 2 --enable-expert-parallel` | 两条 PP 链，各 stage 的专家组跨 DP |

这是接口拓扑说明，不代表每个30B配置都通过性能/质量认证。

`profile_capture.py` 只是可选的合成负载便利封装，默认也全程记录；不必通过它才能 profile。它额外保存运行环境/源码 manifest。`kernel_bench.py` 独立测算子，Graph 算子计时不代表模型支持整模 CUDA Graph。`trace_io.py` 为内部流式读写辅助，不需单独操作。

参考：[PyTorch profiler](https://pytorch.org/docs/stable/profiler.html)、[vLLM profiling](https://docs.vllm.ai/en/latest/contributing/profiling.html)、[Perfetto 文档](https://perfetto.dev/docs/quickstart/trace-analysis)。
