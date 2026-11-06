# Profile：只用三步采集、分析、合并

更新：2026-11-06。以下命令在仓库根目录执行，GPU 采集使用 Linux/CUDA 环境。分析与合并只依赖 Python 标准库，可在没有 GPU 的 Windows 上运行。

## 1. 四个入口各负责什么

| 文件 | 输入 | 输出 | 不负责什么 |
|---|---|---|---|
| `v9/tools/profile_capture.py` | 模型、并行拓扑、合成输入长度/数量、采集窗口 | 每 worker 的完整 Chrome trace、输入、配置、运行日志、源码 SHA256、环境清单 | 不提供可信吞吐；不采 HTTP/媒体前端 |
| `v9/tools/profile_analyze.py` | 同次运行的 `cuda_rank*.json[.gz]` | `summary.json`、中文 `report.md`、八卡 `timeline.html` | 不从 CPU range 猜 GPU 时长；不自动判定空白就是 PP 气泡 |
| `v9/tools/profile_merge.py` | 同机同次运行的多 rank trace | 保留全事件的 `trace.json.gz`，各 rank 的 CPU/GPU/stream 独立显示 | 不逐 rank 归零，不首尾串接；不校准跨机器时钟 |
| `v9/tools/kernel_bench.py` | 算子 shape、dtype、重复次数 | 无 profiler 计时 JSON；可选另采算子 trace | 不测整模型吞吐；Graph 结果不代表引擎有整模 Graph |

`trace_io.py` 是流式读文件的内部辅助，不需要操作。`bench.py offline` 调用 `tools/benchmark.py`，`bench.py serve` 调用 HTTP 压测客户端。`check_logits.py`、`check_generation.py`、`compare_logits.py` 是数值诊断工具，不是 profile 入口。

## 2. 第一份八卡轨迹

```bash
export MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Attention TP8，专家 EP8，共8卡。EP不再是额外设备轴。
python v9/tools/profile_capture.py --model "$MODEL" --output outputs/run_tp8ep8 \
  --tp 8 --pp 1 --dp 1 --enable-expert-parallel \
  --scenario short --requests 32 --prompt-tokens 64 --max-tokens 16 \
  --budget 512 --start-message 1 --messages 8 --record-shapes

python v9/tools/profile_analyze.py --input-dir outputs/run_tp8ep8/trace \
  --output outputs/run_tp8ep8/analysis --expected-ranks 8 --layers 48 --require-nccl --same-host

python v9/tools/profile_merge.py --input-dir outputs/run_tp8ep8/trace \
  --same-host --output outputs/run_tp8ep8/trace.json.gz
```

总 worker 数为 `DP×PP×TP`。采集入口提前检查可见 GPU 数，不足就报错，防止多个 rank 被错误放到同一卡。`--output` 必须是新目录或空目录，避免把不同实验的 rank 混起来。

合成输入直接构造 token IDs，不应用聊天模板，也不以生成内容评价质量。每条输入的实际长度保存在 `manifest.json` 和 `inputs.json`；固定使用贪心采样，忽略 EOS，确保指定的生成长度。当前模型层数是48；分析其他模型时修改 `--layers`。

### 关键参数

| 参数 | 含义 |
|---|---|
| `--requests` | 同时注入的请求总数，不等于每个 GPU forward 的 batch size |
| `--prompt-tokens` / `--long-tokens` | 合成短/长输入的 token 数 |
| `--scenario short/long/mixed` | 全短、全长、或每4条中1条长输入 |
| `--max-tokens` | 每请求生成长度；决定是否有足够 decode 工作 |
| `--budget` | 每调度批次最大 token 预算 |
| `--microbatch-tokens` | PP调度批次预算，默认128；实际还受 `--budget` 约束 |
| `--max-concurrent-batches` | PP在途批次上限，默认 PP size；不是模型虚拟stage数 |
| `--start-message` | 每个 worker 从第几次运行消息开始采，1-based |
| `--messages` | 每个 worker 最多采多少条消息，不是 token 数/请求数/层数 |
| `--record-shapes` | 记录 ATen 输入形状；额外消耗 CPU 和内存 |
| `--max-model-len` / `--max-num-seqs` | 上下文与单批请求上限，默认8192/128 |
| `--max-kv-blocks` | 显式限制 KV 池；根据实际显存设置，太小会触发抢占/重算 |

窗口到达后结束采样；若请求先结束，则在关闭时导出较短窗口。`start-message` 太大可能根本采不到有效消息。长输入的第8条消息可能仍是 prefill，不要仅凭参数认定已经采到 decode。查看 `capture_rankN.json` 的 `is_prefill`、batch ID、请求数，再核对真实 CUDA 层覆盖。扩大窗口会产生很大的八卡文件，并显著干扰调度。

## 3. 八卡流水线和并行对照

分别启动新进程、写不同目录。每行都是8卡，EP开关只改变专家分片方式：

| 方案 | capture 参数 | 看到什么 |
|---|---|---|
| TP8 | `--tp 8 --pp 1` | Attention/专家均TP；TP collective |
| TP8+EP8 | `--tp 8 --pp 1 --enable-expert-parallel` | 一个stage，专家跨8卡dispatch/combine |
| PP2×TP4+EP4 | `--tp 4 --pp 2 --enable-expert-parallel` | rank0–3算前24层，rank4–7算后24层；stage间NCCL P2P |
| DP2×TP4+EP8 | `--tp 4 --pp 1 --dp 2 --enable-expert-parallel` | 两个EngineCore，各自Attention/KV；MoE专家跨DP×TP |
| DP2×PP2×TP2+EP4 | `--tp 2 --pp 2 --dp 2 --enable-expert-parallel` | 两条PP链、每个stage的EP组跨两个DP |

上述是当前接口的拓扑定义和可运行命令，不是全部30B配置都已完成质量/吞吐认证。最近真实采集回归覆盖了四卡TP4+EP4，以及八卡PP2×TP4+EP4的小模型；结果在 [输出索引](../outputs/README.md)。

```bash
python v9/tools/profile_capture.py --model "$MODEL" --output outputs/run_pp2ep4 \
  --tp 4 --pp 2 --enable-expert-parallel --scenario short \
  --requests 32 --prompt-tokens 64 --max-tokens 32 \
  --microbatch-tokens 128 --max-concurrent-batches 2 --messages 12 --record-shapes
python v9/tools/profile_analyze.py --input-dir outputs/run_pp2ep4/trace \
  --output outputs/run_pp2ep4/analysis --expected-ranks 8 --layers 48 --require-nccl --same-host
python v9/tools/profile_merge.py --input-dir outputs/run_pp2ep4/trace \
  --same-host --output outputs/run_pp2ep4/trace.json.gz
```

研究队列填充的对照：保持所有输入/数值后端不变，另开一轮设 `--max-concurrent-batches 1`。不要再用 `lanes=1/2` 代表当前PP并发。两轮分别测无 profiler 的吞吐，再各自采 trace；不能从一次 profile 的墙钟时间报告收益。

### 长度与负载

在同一拓扑下，用新目录重复采集。其他参数保持一致：

```bash
# 长输入。需要足够采集消息才能越过chunked prefill。
python v9/tools/profile_capture.py --model "$MODEL" --output outputs/run_long \
  --tp 8 --enable-expert-parallel --scenario long --long-tokens 4096 \
  --requests 8 --max-tokens 32 --messages 16 --record-shapes

# 混合输入：1/4的请求为长输入。
python v9/tools/profile_capture.py --model "$MODEL" --output outputs/run_mixed \
  --tp 8 --enable-expert-parallel --scenario mixed --prompt-tokens 64 --long-tokens 2048 \
  --requests 64 --max-tokens 32 --messages 16 --record-shapes
```

比较请求数8/32/128、预算128/512/2048。`--requests 128`不保证一个专家一次有128行输入；MoE每个专家的M取决于top-k和路由分布。要在 ATen shape / GEMM kernel 中看实际M。更大的模型上下文并不自动要求写新kernel，先判定注意力读KV、专家小GEMM、主机launch、还是通信主导。

## 4. 输出文件到底是什么

```text
run_pp2ep4/
  manifest.json            精确执行命令、软件版本、源码逐文件SHA、输入SHA、GPU环境、退出码
  config.json              引擎/数值后端配置
  inputs.json              真正传给调度器的token IDs、生成长度、到达step
  results.json             诊断运行及逐请求输出；profile下吞吐和分位数为null
  correctness.json         同拓扑重复一致性；单轮未形成重复对照，不是HF质量认证
  run.log                  完整stdout/stderr，失败时先看这里
  memory_rankN.json        该rank参数/KV与allocator峰值等
  trace/
    cuda_rank0.json ... cuda_rank7.json   真正的CPU+ATen+CUDA+NCCL数据
    capture_rankN.json     采样消息窗口、阶段、CUPTI刷新状态
    audit_rankN.json       基础CUDA事件存在性与类别
    pipeline_rankN.json    仅主机侧run/send/recv调用区间
  analysis/
    report.md               漏卡/漏事件、层归属和重叠复核
    summary.json            热点、逐层、逐消息、API、stream、空隙
    timeline.html           八卡共同时间轴概览；蓝compute、橙NCCL
  trace.json.gz             完整多卡合并文件，供Perfetto打开
```

已整理的样例把 `trace/` 命名为 `raw/`，并无损gzip压缩；内容类别相同。新采集保留原JSON，方便直接导入。

**`pipeline_rank*.json`不是CUDA trace。**它只有Python调用范围，不含ATen或GPU活动，拿它看GPU重叠会得出错误结论。通常只需要 `cuda_rank*` 和完整 `trace.json.gz`。确实要看主机控制图时，单独合并：

```bash
python v9/tools/profile_merge.py --input-dir outputs/run_pp2ep4/trace \
  --kind pipeline --same-host --output outputs/run_pp2ep4/host_calls.json.gz
```

这两类文件的时钟分别为Kineto与host-monotonic，工具拒绝混合。不要把两者任意拼在一个JSON里。

## 5. 不改模型代码，如何知道是哪一层

采集入口显式选择 `ObservedModelRunner` / `ObservedPipelineTransport`。普通模型、EngineCore和ModelRunner不导入profiler。`utils/module_profiling.py` 只在采样窗口挂临时nn.Module hooks，形成：

```text
rank0.stage0.messageN.mbM.prefill / decode   worker这一次模型调用
  stage0.layer0                            第0层完整CPU调用范围
    stage0.layer0.attention                 Attention模块
    stage0.layer0.moe                       MoE模块，含路由/专家/通信
      stage0.layer0.moe.router              Router模块
```

层号是全局decoder层号，从0开始。PP2的stage1从layer24开始；本卡不执行的层不会生成假范围。专家dispatch/combine是内部方法，并非nn.Module，因此没有人为伪造一套独立标签；在MoE范围内结合ATen操作、NCCL和源码辨认。

hooks只提供CPU范围。PyTorch profiler自动记录ATen、CUDA runtime、CUPTI kernel/memcpy/NCCL活动。GPU归层用 **CPU模块范围→CUDA launch correlation/External id→GPU kernel**，不会按GPU事件时间恰好落在CPU条下面就猜归属。异步执行使CPU与GPU条错位是正常的。

父进程开profiler不能自动覆盖子进程，所以每个GPU worker各有一个profiler。当前文件包含**所有被采worker进程的CPU线程与GPU stream**，不包含调度父进程的完整CPU轨迹、HTTP/tokenizer、初始化、模型装载和媒体encoder。观察这些额外阶段需要单独的端到端系统采集，例如 [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)，不能把worker窗口称作完整服务全生命周期。

## 6. 在Perfetto里怎么看八卡并行

1. 打开 [ui.perfetto.dev](https://ui.perfetto.dev/)，选择 `Open trace file`，加载 `trace.json.gz`。
2. 展开 `rankN / stageS / ... CPU pid` 和 `rankN / stageS / ... GPU device`。
3. CPU下是线程、模块range和ATen；GPU下是 `CUDA stream 7/35/...`。stream编号只是运行时标识，不保证7永远计算、35永远通信，必须看实际kernel名称。
4. pin住rank0和rank4的GPU stream，使用同一水平时间窗看stage0/1。它们是多行并列轨迹，共用时间轴，不是只能看一条线。
5. 搜索 `stage1.layer24.moe` 找CPU范围，再看该范围发起的CUDA API和flow；搜索 `nccl` 找真正的GPU通信kernel。

合并工具给pid/tid、correlation和flow ID加rank命名空间，保留CPU↔GPU关联；按 `baseTimeNanoseconds` 对齐，再减**共同**起点。不会分别把每张卡最早事件设0，也不会把八卡串行接在后面。跨卡时间先后保留，但不同worker独立开启窗口带来的边界偏差仍存在。

### 什么才是计算通信重叠

同一卡上，一个stream的GEMM和另一个stream的NCCL **GPU执行区间在横轴相交**，才是该卡观察到的计算通信并发。不同stream仅表示可能并发；CPU的异步API返回早、CPU range互相覆盖，都不是证据。

设计算区间并集为C，NCCL区间并集为N，报告计算 `|C∩N|` 与 `|C∩N|/|N|`。区间并集避免嵌套/重叠重复计时。NCCL内部可能等待peer，因此交集不等于有效带宽完全被掩盖，也不说明SM资源没有相互争用。

跨stage计算交集是另一项：stage0在GPU0计算批B，同时stage1在GPU4计算批A，这是PP流水。即使每张卡内部NCCL与计算完全串行，跨stage仍可并行。不要混淆两种重叠。

同一CUDA stream中的kernel按序执行；同一GPU进程可以向多个stream提交kernel，硬件在资源/依赖允许时并发执行。同一CPU线程不会同时运行两条Python指令，但它可以提前提交多个异步GPU任务。

### 空白与PP气泡

先缩小到双方GPU都在采样、队列有独立请求的稳态区间。看上游是否及时生产activation、recv是否等待、CPU是否停在同步/IPC、下游是否被MoE长尾卡住。再比较相同输入下在途深度1与PP size的trace和无profile吞吐。

采样起停、窗口漏尾、主机launch供给不足、抢占/重算、NCCL等peer、请求本身缺少并行度，都可产生空白。`observed_inactive_fraction`只是已记录GPU窗口中的无活动比例，不能直接写成PP bubble率或SM idle率。CUDA trace也不提供SM利用率/occupancy；这些需硬件计数器或Nsight Compute。

## 7. 为什么打开是空的，怎样验收

先用 `profile_analyze.py`，不要先相信截图。它检查rank集合、ATen/CUDA runtime/kernel/模块标签、全局层覆盖；`--require-nccl`再检查GPU NCCL事件。没有CUDA时返回失败。`capture_rank`/audit通过仍不能证明每条消息的最后一层都被捕获，分析会额外列出逐消息覆盖警告。

常见原因：打开了host-only的 `pipeline_rank`；用了只含record_function的旧简化文件；profiler采错进程；窗口没命中；选到 `.partial`；CUPTI没导出GPU buffer；gzip或内存问题。全文件很大时，优先看小模型完整样例，或选择同次运行两个rank合并，不裁剪它们的事件：

```bash
python v9/tools/profile_merge.py --same-host \
  --inputs outputs/profiles/qwen_pp2ep4_legacy/raw/cuda_rank0.json.gz \
           outputs/profiles/qwen_pp2ep4_legacy/raw/cuda_rank4.json.gz \
  --output outputs/two_ranks.json.gz
```

Perfetto不识别gzip时可用Python `gzip` 解压后打开；解压后的八卡JSON可能超过1GB。`timeline.html` 是像素分桶概览，不含CPU/算子详情；精确微秒重叠必须回到完整trace。

CUDA13环境出现过最小矩阵乘法也只导出CPU的情况。设备同步只保证GPU完成，不保证CUPTI活动缓冲已交给Kineto；独立采集适配器在stop前同步并调用 `cuptiActivityFlushAll` 兼容处理，保存状态码，再实际检查kernel事件。根因只能确定在该环境的活动缓冲交付链路，不能据此断言所有PyTorch/CUDA版本都有同一缺陷。失败状态不被包装成有效GPU采集。

## 8. 吞吐和P95/P99另跑

```bash
python v9/bench.py offline --model "$MODEL" --output outputs/bench_pp2ep4 \
  --tp 4 --pp 2 --enable-expert-parallel --workloads short128,mixed64,context4k8 \
  --budget 512 --microbatch-tokens 128 --max-concurrent-batches 2 --repeats 3
```

该入口先warmup，再保存逐请求输出、首token/完成时间和P50/P95/P99；不传 `--profile`。它测本地engine完成事件，不包括网络/tokenizer。分位数使用排序观测值 `round((n-1)×q/100)`，小样本P99不能代表生产长尾。

要复用profile的**相同输入**，传 `--inputs outputs/run_pp2ep4/inputs.json --workloads short`，并固定相同并行、budget和microbatch参数。不要拿默认short128与另一个32请求profile冒充同负载。

HTTP测试使用两个终端：

```bash
# 终端1：正常服务，吞吐实验不加--profile。
python v9/server.py --model "$MODEL" --tensor-parallel-size 8 --enable-expert-parallel --port 8000
# 终端2：closed-loop客户端，256请求、32并发。
python v9/bench.py serve --base-url http://127.0.0.1:8000/v1 \
  --output outputs/http_bench --requests 256 --concurrency 32 --max-tokens 32
```

HTTP记录包括逐请求成功/失败、输出token数、E2E/内容TTFT/TPOT的分位数及chunk ITL。SSE chunk不等于token；缺少usage时不伪造tok/s。`bench serve --profile`只把客户端负载标记为诊断；采GPU还需服务端 `--profile`，详见 `server.py --help`。服务端profile只包worker，仍不会自动采到HTTP主线程。

参考：[PyTorch profiler](https://pytorch.org/docs/stable/profiler.html)、[Perfetto入门](https://perfetto.dev/docs/quickstart/trace-analysis)、[vLLM profiling](https://docs.vllm.ai/en/latest/contributing/profiling.html)。
