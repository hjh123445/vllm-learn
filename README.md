# vllm-learn：从零手搓 LLM 推理引擎

六个版本把一个推理引擎的核心机制逐层造出来：静态批处理 → 迭代级调度（Orca 混合批次）
→ PagedAttention 显存管理 → nano 调度栈（APC/chunked prefill）→ 手写 triton kernel 层 → 在线推理服务。
**每一步都有 benchmark 数字和正确性裁决，结论全部实测**（方法学 `benchmark/README.md`，
思考过程 `benchmark/PERFORMANCE_NOTES.md`，每版详解 `vX/README.md`）。

- 正确性：**fp32 下与 HuggingFace transformers 逐 token 100% 对齐**（v2/v3/v4 三版都验过）
- 性能：v4 吞吐 405.6 tok/s（Qwen3-0.6B），超过 vLLM eager 模式、达其 graph 模式 56%–70%

## Quick Start

环境细节（flash-attn 编译 / vLLM 统一环境 / 代理坑 / GPU 共享）见 **[ENVIRONMENT.md](ENVIRONMENT.md)**；
全程踩坑与排查过程（⭐ 面向面试的深度复盘）见 **[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)**。

```bash
pip install -r requirements.txt          # 依赖清单（含版本实测记录）

# 模型：默认在仓库根的上一级找 Qwen3-0.6B，可用环境变量覆盖
export VLLM_LEARN_MODEL=/path/to/Qwen3-0.6B     # 或 Qwen3-8B（36 层 / GQA 4:1，已兼容）

# 1) 跑一个版本（每版都是独立完整的引擎）
cd v4 && python example.py                      # 批量推理：4 条 prompt，流式打印
cd v5 && python server.py --port 8000 &          # 在线服务：OpenAI 兼容 + SSE 流式
curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'

# 2) 验证正确性（14/14 PASS 应全绿）
python benchmark/accuracy.py                     # 确定性/batch 一致/跨版本等价/HF fp32 对齐

# 3) 跑性能
python benchmark/speed.py --engine v4            # TTFT/TPOT/吞吐
python benchmark/serve_bench.py --url http://127.0.0.1:8000 --num-requests 64   # 服务级压测
```

## 版本文档（每个版本一份详细 README：代码结构 + 技术原理 + 实现 + 踩坑）

| 版本 | 主题 | 文档 |
|---|---|---|
| v0 | 静态批处理（左 padding / 加性 mask / 装载对账） | [v0/README.md](v0/README.md) |
| v1 | 迭代级调度 + Orca 混合批次（平铺批 / 逐序列朴素注意力） | [v1/README.md](v1/README.md) |
| v2 | PagedAttention + 显存统一管理（warmup 预算 / 抢占 / Context） | [v2/README.md](v2/README.md) · [v2/DESIGN.md](v2/DESIGN.md) |
| v3 | nano-vllm 调度栈原样搬入（APC / chunked prefill / 引用计数） | [v3/README.md](v3/README.md) |
| v4 | 手写 triton kernel 层（融合算子 / FlashAttention / guard 税策略） | [v4/README.md](v4/README.md) |
| v5 | 在线推理（异步引擎 / OpenAI SSE / 优雅退出） | [v5/README.md](v5/README.md) |
| v6 | 多卡并行（TP/EP/PP 及组合，单机矩阵+跨节点实测） | [v6/README.md](v6/README.md) · 小版本: [v6/MILESTONES.md](v6/MILESTONES.md) |
| v7 | 投机解码（n-gram draft + verify，正确性 3/3） | [v7/README.md](v7/README.md)（含 v7.0-v7.3 小版本） |
| v8 | 上下文并行 CP/DCP（LSE 数学 + DCP2 端到端） | [v8/README.md](v8/README.md)（含 v8.0-v8.4 小版本） |

## 版本演进

| 版本 | 核心机制 | 关键文件 |
|---|---|---|
| **v0** | 静态批处理：Batch 组好跑到整批完成，左 padding + mask | `v0/engine/`（每版 README 有完整结构） |
| **v1** | 迭代级调度 + **Orca 混合批次**：prefill 和 decode 摊平在同一个 forward 里跑（flattened batch，逐序列朴素注意力） | `v1/engine/` |
| **v2** | **PagedAttention**：16-token block + block table + warmup 实测池子预算 + 按 block 准入 + 抢占（recompute）+ attention 上下文收进全局 Context | `v2/engine/{kv_cache,context}.py` |
| **v3** | nano-vllm 的调度栈**原样搬入**（APC 前缀缓存 + block 引用计数 + chunked prefill + max_num_batched_tokens）+ 我们的模型；重写 model_runner（单卡单进程）/llm_engine | `v3/nanovllm/engine/` |
| **v4** | **全手写 triton kernel 层**：融合算子（残差+RMSNorm、q/k norm+rope+cache 写入、SiLU×up）+ 手写 paged FlashAttention（decode 在线 softmax / prefill causal）+ torch.compile 只用于采样器 | `v4/layers/{kernels,flash_attn}.py` |
| **v5** | **在线推理**：异步引擎（后台循环线程 + 每请求流式队列）+ stdlib HTTP 服务（OpenAI 兼容 SSE 流式），请求随到随入（并发 4 路流式中 TTFT 16-23ms，中途新请求 24ms 立即准入） | `v5/server.py`、`v5/engine/async_engine.py` |

每一版都是独立可运行的完整引擎（`python example.py`），模型加载不依赖 transformers（tokenizer 用它）。

## 性能矩阵（2026-10-27 实测）

负载与口径：12 请求、并发 8、max_tokens 32/96/160 混合、greedy、bf16；
单卡 A800-80G；数据在 `benchmark/results/perf_matrix.json`。

### Qwen3-0.6B（28 层 / hidden 1024 / GQA 2:1 / tie embedding）

| | v0 | v1 | v2 | v3 | **v4** | nano-vllm graph（参照） |
|---|---|---|---|---|---|---|
| TTFT avg | 4463 | 717 | 560 | 117 | **26.5 ms** | 1207 |
| TPOT avg | 17.9 | 35.5 | 33.8 | 39.0 | **13.4 ms** | 4.1 |
| 吞吐 | 155.6 | 146.8 | 145.1 | 156.4 | **405.6 tok/s** | 581.6 |

### Qwen3-8B（36 层 / hidden 4096 / GQA 4:1 / untied）

| | v0 | v1 | v2 | v3 | **v4** |
|---|---|---|---|---|---|
| TTFT avg | 2608 | 946 | 708 | 142 | **30.7 ms** |
| TPOT avg | 25.2 | 49.5 | 43.6 | 48.9 | **20.6 ms** |
| 吞吐 | 141.5 | 106.1 | 117.5 | 124.8 | **270.4 tok/s** |

读表要点（全部有 profile 依据，详见 PERFORMANCE_NOTES）：

- **v0/v1/v2/v3 的 decode 都是 CPU 发射瓶颈**（0.6B 时 GPU 87% 空闲）——它们 TPOT 同量级，
  区别主要在 TTFT（调度方式）和吞吐（GPU 利用率）；
- **v4 的手写 kernel 把 TPOT 打下来 2.6 倍**：融合算子 + flash attention 让 kernel 数/层从
  ~25 降到 ~10；TTFT 从 117ms 到 26ms 是 prefill 从逐序列循环变单 kernel；
- 8B 上 v4 优势收窄（13.4→20.6ms）是符合预期的：模型越大 GPU 越忙，CPU 发射的相对占比越小；
- v3 的 8B 池子用 0.5 利用率（0.3 下 warmup 峰值就吃满预算，触发抢占风暴——
  **大模型要按权重+激活峰值重新选 util**，这是量出来的教训）。

### 与正经 vLLM 的服务级压测（Qwen3-0.6B，128 请求 × 128 tok，20ms 到达流）

| | v5（我们，在线服务） | vLLM 0.31 graph | vLLM eager |
|---|---|---|---|
| TPOT | 38.8 ms | **2.8 ms** | 18.2 ms |
| 吞吐 | 1800 tok/s | **5551 tok/s** | 1602 tok/s |
| TTFT avg | 20.6 ms | 14.2 ms | 3021 ms（排队） |

归因（详见 PERFORMANCE_NOTES §9）：vLLM 的 decode **已打满显存带宽**（权重 1.2GB ÷ 2.8ms
≈ 430GB/s ≈ A800 HBM），batch 摊薄权重读取，TPOT 与 batch 无关；我们仍是 CPU 发射瓶颈。
两个 eager 引擎吞吐同量级（1800 vs 1602），**CUDA graph 一项 = 3 倍吞吐 / 13.9 倍 TPOT**。
工具：`benchmark/serve_bench.py`（通用流式压测，任意 OpenAI 兼容端点，同为 v5 server / vllm serve 服务）。

### 与 nano-vllm 对照（同负载，0.6B）

| | 我们 v4 | nano eager | nano graph |
|---|---|---|---|
| TPOT | 13.4 ms | 48.2 | 4.1 |
| 吞吐 | 405.6 | 108.9 | 581.6 |

v4 ≈ nano eager 的 3.7 倍、nano graph 的 70%。剩余差距全部是 CUDA graph（CPU 发射），
已由 profile 定量（triton launch 自带 cudaFuncSetAttribute ~15µs/次 × ~280 次/步）。

## 正确性体系（`python benchmark/accuracy.py`，当前 14/14 PASS）

| # | 测试 | 裁决 |
|---|---|---|
| T1 | greedy 确定性 | 同请求两次输出逐 token 相同 |
| T2 | batch 一致性（fp32） | batch 里跑 == 单独跑（抓 padding/mask/位置错误） |
| T3 | v0/v1/v2/v3 等价（bf16 单请求路径） | 重构无回归（v4 由 T4-fp32 把关：其 triton kernel 全程 fp32 计算，比 bf16 torch 路径更精确） |
| T4 | HF transformers 对齐（fp32） | 与 HF generate 逐 token 一致：v2/v3/v4 均 **32/32 = 100%** |

## Benchmark 体系（`benchmark/`）

| 脚本 | 类别 | 说明 |
|---|---|---|
| accuracy.py | 正确性 | 上表四个测试 |
| speed.py / harness.py | 速度 | TTFT/TPOT/E2E/吞吐；worker 子进程 + sys.path 注入调度各引擎 |
| capacity.py | 容量 | 显存账本/池子/并发容量（v2 起 paged：同显存并发 16 → 203 条 1024-token 请求） |
| stress.py | 压力 | 到达流 + 时延分位数（v1/v2 的 TTFT p99 304 → 47ms） |
| bubble.py | 可视化 | 静态批处理气泡 Gantt 图 |
| nano.py | 外部对照 | 同负载跑 nano-vllm（graph/eager 两组） |
| profile_pipeline.py | 归因 | torch.profiler + chrome trace |
| serve_bench.py | 服务级压测 | 通用流式压测（任意 OpenAI 端点）：TTFT/TPOT/E2E 分位数/吞吐 |

模型路径：`VLLM_LEARN_MODEL` 环境变量（默认仓库根的上一级找 `Qwen3-0.6B`）。
性能矩阵复现：

```bash
for m in Qwen3-0.6B Qwen3-8B; do
  for e in v0 v1 v2 v3 v4; do
    VLLM_LEARN_MODEL=../$m python benchmark/speed.py --engine $e --num-requests 12 --batch-size 8
  done
done
# 8B 的 v3/v4 记得加 --gpu-mem-util 0.5（0.3 的预算装不下 warmup 峰值）
```

## 目录

```
v0/  静态批处理                v3/  nano 调度栈 + 我们的模型
v1/  迭代级调度 + 混合批次       v4/  + 手写 triton kernel 层
v2/  PagedAttention + Context   v5/  + 在线推理（server.py / async_engine）
benchmark/  测试体系 + 结果 JSON/图表
```

## 在线推理（v5）

```bash
cd v5 && python server.py --model ../Qwen3-0.6B --port 8000 --gpu-mem-util 0.3
curl http://127.0.0.1:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'  # SSE 流式
```

线程模型：HTTP 线程 tokenize→入队→从流式队列读 token；引擎循环线程（唯一写者）schedule→run→postprocess→
推增量。队列安全依据：scheduler 的 deque append/popleft 在 CPython 原子（单写多读无竞态）。

## 环境

torch 2.9.1+cu130 / triton 3.5.1 / flash-attn 2.8.4（源码编译，sm80，nano 对照用）/
A800-80G / 单卡单进程。`v2/DESIGN.md` 是 paged 设计文档；各提交信息含完整的决策与踩坑记录。
