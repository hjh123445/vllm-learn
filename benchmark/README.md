# Benchmark 体系

测三件事：**算得对不对**（accuracy）、**算得多快**（speed）、**时间花在哪**（profile）。
对 v0（静态批处理）和 v1（迭代级调度）用同一套方法学。

```
benchmark/
├── README.md           # 本文档：方法学
├── harness.py           # 计时 harness：对 v0/v1 复刻带时间戳的生成循环，产出记录
├── accuracy.py          # 正确性测试套件
├── speed.py             # 速度测试：TTFT / TPOT / 时延 / 吞吐
├── profile_pipeline.py  # 推理链路 profile：torch.profiler + 阶段计时
├── bubble.py            # 流水线气泡可视化：v0 vs v1 的请求时间线 Gantt 图
└── results/             # 跑出来的图表和 JSON
```

---

## 0. 为什么测试脚本要"复刻"生成循环

v0 的 `ModelRunner.run()` 和 v1 的 `Engine.generate()` 是黑盒，拿不到每个请求每个迭代的
时间戳。所以 `harness.py` 不走 `Engine.generate()`，而是直接调用组件
（`Scheduler` + `ModelRunner`）复刻主循环，在关键点打时间戳。

这本身就是对分层设计的检验：主循环只是一个 10 行的 for，复刻它不等于复制实现。
代价是 harness 和引擎循环有重复，改引擎循环时要记得同步 harness。

## 1. 正确性测试（accuracy.py）

推理框架的"对"没有绝对参照，我们用三层递进的判据，从便宜到贵：

| # | 测试 | 判据 | 抓什么 bug |
|---|------|------|-----------|
| 1 | greedy 确定性 | 同一请求跑两次，输出逐 token 相同 | 采样路径的状态泄漏（cache 未清、随机数未控） |
| 2 | batch 一致性 | 4 条不同长度 prompt：batch 里跑 vs 单独跑，greedy 输出相同 | **padding / mask / 位置编码错误**——最经典的批处理 bug。一个请求在 batch 里和单独跑走的是不同的张量路径，算错必现分歧 |
| 3 | v0 vs v1 等价 | 同一批请求，两版单请求路径输出相同 | 重构引入的逻辑偏差（attention 重写、cache 行号索引） |
| 4 | HF 对齐 | greedy 前 N 个 token 与 transformers 的 generate 相同 | 模型实现本身（RMSNorm/RoPE/GQA/MLP 的任何一处写错） |

用法：
```bash
python benchmark/accuracy.py              # 全部测试
python benchmark/accuracy.py --test 2    # 只跑测试 2
```

**已知边界（不是 bug）**：bf16 下 batch 形状不同，matmul kernel 的归约顺序不同，
长序列的 greedy 输出可能在几十个 token 后分歧（实测：单请求路径两版逐位一致，
4 条合批时 2/4 条在第 23/43 个 token 分歧）。所以：
- 测试 2/3 用**单请求或相同形状**路径对比；
- 对齐 HF 时比较前 32 个 token 的前缀一致率，不要求全序列。

这是所有 bf16 推理引擎的共性（vLLM 同 batch 不同形状也会有 1e-2 级 logits 差），
fp16/fp32 下差异会缩小。要做逐位回归，把 dtype 换 float32 再跑。

## 2. 速度测试（speed.py）

### 指标定义

| 指标 | 定义 | 反映什么 |
|------|------|---------|
| TTFT | 请求发出 → 第一个 token 产出（prefill 完成时刻） | 用户体感"首字时延" |
| TPOT | (完成时刻 - 第一个 token 时刻) / (生成 token 数 - 1) | 流式输出的每个 token 间隔 |
| E2E | 请求发出 → 完成 | 单请求总时延 |
| 吞吐 | 输出 token 总数 / 总墙钟时间 | 系统产出能力 |

### 控制变量（不改这些，数字没有可比性）

1. **greedy（temperature=0）**——采样耗时微小但排除随机性更干净；
2. **dtype 固定 bfloat16**——换精度数字没有可比性；
3. **GPU 空闲时跑**，先跑一次小 warmup（kernel JIT/cuBLAS 选核）再计时；
4. **同一 checkpoint、同一 prompt 集**（脚本内置固定 prompt，保证可复现）；
5. 记录 batch_size 和 max_tokens 分布——静态批处理的数字强依赖这两个。

### 负载设计

对比 v0/v1 必须用**混合长度负载**：如果所有请求生成长度一样，v0 没有尾部浪费，
v1 的优势显示不出来。内置负载：N 条请求，max_tokens 在 {短, 中, 长} 挡位轮转
（默认 32/96/160），模拟真实流量里"有的问题一句话答完、有的长篇大论"。

### 用法
```bash
python benchmark/speed.py --engine v0 --num-requests 12 --batch-size 8
python benchmark/speed.py --engine v1 --num-requests 12 --batch-size 8
```
结果打印 TTFT/TPOT/E2E/吞吐 并存 JSON 到 `results/`。

### 预期结论（帮助校验实现没写反）

- v0：短请求的 E2E 被同 batch 的长请求拖住（TPOT × 长请求 token 数）；
- v1：短请求按时完成并立刻腾出 slot，总吞吐高、长尾时延低；
- v1 的 TTFT 在"请求陆续到达"场景下显著更低（slot 随时回收随时准入）。

## 3. 推理链路 profile（profile_pipeline.py）

两个层次：

### 3.1 阶段计时（宏观：时间花在哪个阶段）

harness 在每个关键点打时间戳，汇总出：prefill 总时间 / decode 总时间 / 采样总时间 /
mask 构造总时间（mask 在 CPU 上构造，是已知的热点候选）。这层回答"慢在哪个阶段"。

### 3.2 算子级 profile（微观：阶段里哪个算子慢）

`torch.profiler` 包住整个生成循环，导出 Chrome trace：
```bash
python benchmark/profile_pipeline.py --engine v1 --num-requests 8 --max-tokens 32
```
- 终端打印按 CUDA 时间排序的 top 算子表；
- Chrome trace 存 `results/trace_{engine}.json.gz`，浏览器打开
  `chrome://tracing` 加载，可以看到每个 kernel 的时间线（含 kernel 间隙 = GPU 空闲）。

### 怎么读 trace

- 看 **CPU 行和 GPU 行的时间对齐**：decode 循环里如果 GPU kernel 之间有大量空白，
  说明 CPU 端（python 循环、mask 构造、逐行写 cache）跟不上，GPU 在等 CPU——
  v1 的 `KVCache.write` 逐行 for 循环和 `read` 的高级索引拷贝是主要嫌疑；
- 看 `aten::index/index_select` 占比：就是 cache gather；
- 看 softmax 和两个 matmul（QK^T、PV）的形状是否符合预期
  （decode 时 [b,h,1,L] @ [b,h,L,hd]）。

## 4. 气泡可视化（bubble.py）

### 什么是气泡

静态批处理的气泡：batch 里短请求先完成，但 slot 不释放，**GPU 继续为已完成的请求
算 256 维的 logits 和整层前向**，直到最慢的请求结束。这段时间 GPU 在烧算力，
产出为 0。

### 怎么可视化

bubble.py 用 harness 跑同样的负载（v0 和 v1 各一次），记录每个请求的
[等待, prefill, 生成, (v0: 已完成仍占位)] 时间区间，画成 Gantt 图：

- 每行一个请求，横轴是墙钟时间；
- 蓝色段 = 有效生成；灰色段 = 等待准入（v1 才有）；红色斜线段 = **气泡**
  （已完成但仍占着 slot 的死时间）；
- 附一条 batch 利用率曲线（活跃请求数 / batch_size 随时间变化）。

```bash
python benchmark/bubble.py --num-requests 12 --batch-size 8
# 输出 results/bubble_v0_vs_v1.png + results/bubble_summary.json
```

读图要点：v0 的红色斜线面积就是静态批处理浪费的算力；v1 的利用率曲线应在短请求
完成后立刻掉下来再被新请求填回去。

## 5. 结果归档

所有脚本把数字存 JSON、图存 PNG 到 `benchmark/results/`（trace 文件较大，已 gitignore）。
对比结论写进 commit message，原始数字进 JSON，方便以后回溯性能回归。

### 5.1 实测结果（2026-10-25, Qwen3-0.6B bf16, 12 请求 / batch_size=8 / max_tokens 32/96/160）

| 指标 | v0 静态批处理 | v1 迭代级调度 | 说明 |
|------|--------------|--------------|------|
| TTFT avg / max | 1966 / 5546 ms | 1115 / 3756 ms | v0 第二批请求要等第一批全部完成才能 prefill；v1 slot 随收随放 |
| TPOT avg | 36.4 ms | 49.4 ms | v1 的 cache gather / 逐行写 / .item() 同步更重（见 profile） |
| 总墙钟 | 10.4 s | 9.7 s | v1 略胜 |
| 吞吐 | 111 tok/s | 119 tok/s | 同上 |
| **slot 时间浪费率（气泡）** | **35%** | **0%** | v0 完成即占位到整批结束；bubble_v0_vs_v1.png 红色斜线段 |

诚实的结论：**v1 消灭了气泡（slot 利用 42.8s 有效 vs v0 占用 65.8s 只有 42.8s 有效），
TTFT 明显改善，但总吞吐只小胜**——因为 0.6B 模型 decode 在 A800 上是 CPU 发射瓶颈
（profile 实测 Self CPU 6.0s vs Self CUDA 0.76s，GPU 87% 时间在等 CPU），
v0 那些"浪费"的 batch 行几乎是免费的；v1 反而背了 gather 的开销。
GPU 越忙（模型越大、batch 越满），v0 的气泡成本越高，v1 的优势越大——
换更大的模型或加重计算负载可以验证这个推论。

## 6. 已知问题与 TODO

- 采样参数整批共用第 0 条请求的（v0/v1 都这样），做 per-request 温度采样前，
  带温度的吞吐数字不严谨（greedy 不受影响）；
- v1 `KVCache.read` 的高级索引每层每步拷贝 [b, kvh, max_len, hd]，profile 里会看到
  明显的 index 开销，v2 用 paged KV / 统一变长布局消掉；
- v1 prefill 是逐条 batch=1，多请求同时到达时 prefill 吞吐低于 v0 的合批 prefill
  （trade-off：实现简单、无 padding；v1.1 可以加合批 prefill）。
