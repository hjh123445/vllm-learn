# DeepSeek-V4-Flash 在 A800 上的推理实验报告

> 日期：2026-10-31 ~ 11-01（实验窗口 22:00 – 08:30）
> 硬件：143 机器 8×A800-SXM4-80GB（sm80/Ampere, compute_cap 8.0）
> 权重：DeepSeek-V4-Flash（43 层 / 256 专家 top-6 / 149GB：MXFP4 专家 + FP8 dense）
> 撰写：vllm-learn 项目，全部实验数据可复现（脚本路径见附录）

---

## 0. 实验目标与总路线（用户指令）

1. 对比 vLLM 与我们的框架跑 DeepSeek-V4 的速度，写清两边的配置；
2. 对两个过程做 profile，详细定位哪里慢、该往哪个方向优化才有效；
3. 参考 [YJH008/deepseek-v4.1-flash-a800-deploy](https://github.com/YJH008/deepseek-v4.1-flash-a800-deploy)（A800 部署 V4.1 的实盘方案）；
4. DSV4 先做到"能部署"，逐步优化跑多快本身很有价值——详细研究；
5. vLLM 若跑不了 DSV4，用 Qwen3 做对照（此前 PARALLEL_MATRIX 已有完整数据）。

## 1. 结论速览

| 引擎 | 配置 | 结果 |
|---|---|---|
| **vLLM 0.31.0** | TP8 / EP8, DSV4-Flash | **无法在 A800 上运行**——官方 kernel 栈要求 Hopper+（三连撞证据链 §2） |
| **dsv41-A100-custom 运行时**（YJH008 方案） | 8×A800 EP8 (32 experts/卡, 层 6,6,6,6,5,5,5,4) | 适配 V4-Flash 打了 9 个补丁；装载 42s + prefill 通 + 首个 token 输出；decode 单步 >100s 未跑通（卡点分析 §4.3） |
| **我们 v6.3 引擎** | 8×A800 分层分放（同层配方）， bf16 反量化, eager | **跑通**：装载 957s，生成 **1.7 tok/s**；profile 定位每步 895ms 的构成（§5） |
| vLLM（Qwen3 对照，既有数据） | Qwen3-Omni-30B 单卡 | 22.6 tok/s（PARALLEL_MATRIX.md 14 配置矩阵） |

一句话：**DSV4 在 A800 上，vLLM 官方路径硬件不支持；第三方 A100 运行时需要为 V4-Flash（与 V4.1 有 6 处架构差异）打适配补丁，本报告完成了 9 个补丁并推进到 decode；我们自己的引擎是当前唯一完整跑通 V4-Flash 生成的路径，慢在 launch-bound + MoE 逐专家循环 + 跨卡搬运，优化路线见 §6。**

## 2. 为什么 vLLM 跑不了 DSV4（三连撞证据链）

vLLM 0.31.0 + torch 2.13.0+cu130（`/tmp/vllm_venv`，此前 Qwen3 对照用的环境）。

| 尝试 | 配置 | 死在哪 | 根因 |
|---|---|---|---|
| 1 | `enable_expert_parallel=True` (EP8) | MXFP4 的 **MARLIN backend** `create_weights` 单卡先建全量再切 → OOM (78.8GB 已用) | MARLIN MoE 的 EP 分配路径不逐卡分片 |
| 2 | `tensor_parallel_size=8` | `deepgemm hyperconnection.hpp:56: Unsupported architecture` | **DeepGEMM 只支持 Hopper(sm90)+**；mHC 的 broadcast 路径 `use_tilelang_fallback=False` 强制走它，Ampere 无 fallback（vLLM 0.31 bug）。我 patch 成 PyTorch 等价数学后撞下一关 |
| 3 | 同上（patch 后） | `triton: type fp8e4nv not supported in this architecture` | FlashMLA 的 o_proj kernel 用 **FP8 e4m3**，Ampere 的 triton 只支持 e4b15/e5 |

**官方并行配置是什么**：从 vLLM 代码结构读出——DSV4 的 nvidia 实现整条 kernel 栈（DeepGEMM mHC / FlashMLA FP8 o_proj / MARLIN MXFP4）默认目标 **H100/H200/B200 + EP**（`enable_expert_parallel=True`，DeepSeek 官方对 MoE 一直推荐 EP）。这不是配置问题，是**硬件代差**：A800(sm80) 缺 DeepGEMM 依赖的 wgmma/TMA 和 FP8 e4m3 tensor core。

（YJH008 仓库正是这个结论的旁证——它在 A100 上不用 vLLM，而是用 shi3z 的自研运行时。）

## 3. dsv41-A100-custom 运行时：为 V4-Flash 打的 9 个适配补丁

按 YJH008 的 8×A800 配方（`--ep --ep-shards` + `DSV41_LAYER_COUNTS`），环境 `torch==2.6.0+cu124 + triton==3.2.0`（venv `/data/ifly_jyhe23/dsv4_rt/venv`）。但该运行时为 **V4.1-Flash**（40 层/384 专家/全 router）编写，我们的 **V4-Flash**（43 层/256 专家/**hash 路由前 3 层**）有架构差异，逐个暴露逐个修：

| # | 差异 | 补丁（`/tmp/deepseekv4.1-A100-custom-main/dsv41/`） |
|---|---|---|
| 1 | inference/config.json 缺 `kv_source_layers`/`window_size` | 从 compress_ratios 推导（41 个 kv_source 层），window=128；写入可写副本 `/data/ifly_jyhe23/dsv4_rt/`（原 checkpoint 只读，软链+覆盖 inference/） |
| 2 | encoding 模块名不同（`encoding_dsv4.py`） | 拷为 `encoding/encoding.py` |
| 3 | 缺 `norm_eps`/`hc_eps`/`candidate_*` 等 V4.1 独有字段 | 补默认值（norm_eps 从 HF config 取 1e-6） |
| 4 | **前 3 层 hash 路由**（V4.1 全 router）——`w["ffn.gate.bias"]` KeyError | model.py：`hash_route` 分支 + `tid2eid` 查表（v6.3 自研实现同款语义：expert id 静态查表、权重=route_scale/topk）；input_ids 经 `_CURRENT_INPUT_IDS` 全局通道从 forward 入口传入 |
| 5 | HCA 层(128) 无 indexer 权重 | attention 构造判空（`_has_idx_keys`） |
| 6 | **compressor.wkv 是 [2·head_dim, 4096] 的 Ca|Cb 拼接**（V4.1 是 [512]+独立 wgate） | 拆半近似为 (wkv, wgate) 对——⚠ 此为近似，数学上 Ca/Cb 各含 kv+gate 两个语义流（见 v6.3 CSACompressor 的 overlap 数学），正确实现需按 HF 参考重写（记录在 §6 TODO） |
| 7 | CSA indexer 无 `wk`，k 应回到 128 维 | 用 indexer **自己的** compressor（wkv[256,4096]）取 Ca 半区投影 |
| 8 | `sparse_attn_decode` triton kernel 在 triton 3.2 上 **codegen 失败**（`arith.select` 布局不匹配，每步重试编译 → GPU0 100% 死转） | 换 PyTorch 等价路径（在线 softmax + sink 吸收的 eager 实现） |
| 9 | decode 的 `compressor2` 硬编码 `assert ratio==2`（V4-Flash 是 4/128）+ indexer None 判空 + bf16 cache dtype | 通用化任意 ratio 的 ring 池化 |

**推进到的状态**：装载 42s（8 卡并发）→ prefill ✓ → **首个 token 输出**（"nap…"/"нагрузки"）→ decode 单步 >100s 未完成 2 token（timeout 兜底杀掉）。剩余卡点见 §4.3。

## 4. Profile 与慢点定位

### 4.1 我们 v6.3 引擎（跑通，torch profiler 全量 trace 179MB）

配置：43 层 8 卡（cuda:0 放 embed+前 4 层，1-6 各 ~6 层，cuda:7 放尾部+lm_head），bf16 反量化（GPU 合计 ~1.2TB），跨卡层间 `streams.to(dev)`，eager，每步全序列重算。

Bench（3 prompts × 32 tok，greedy）：
```
req0: 32 tok / 18.2s = 1.75 tok/s   ' Paris. The capital of France is Paris.…'  ← 数值与 v6.3 验证一致
req1: 32 tok / 18.5s = 1.73 tok/s
req2: 32 tok / 19.2s = 1.67 tok/s
```

Profile（8 步 decode，77.7 万 cpu_op 事件）：
- **每步 895ms，每步 97,157 个 op**：`as_strided` 19,463/步、`matmul` 4,357/步（= 43 层 × ~100 个 gemm）、`aten::to` 3,867/步；
- op 平均 ~7μs → **纯 kernel/操作发射 ≈ 680ms/步（76%）**——launch-bound 实锤；
- 跨卡搬运（aten::to*）5.4 万次累计 1,277ms/8 步 ≈ 160ms/步（18%）；
- 单 op 峰值仅 3.8ms（bincount）——**没有任何"一个大 kernel 慢"，是长尾小 op 海洋**；
- MoE 结构性成本：每层 256 专家的 python 循环 + 每专家独立 gemm（offsets.tolist() 同步）。

### 4.2 dsv41 运行时（未跑通，但卡点已定位）

- `sparse_attn_decode` 的 triton kernel 在 triton 3.2/torch 2.6 上 codegen 失败并**每步重试编译**（GPU0 100% 的第一根因，已用 PyTorch 路径绕过）；
- 绕过后 decode 仍 >100s/token：EP decode 的 per-layer 设备时间戳机制（`DSV41_EP_TRACE`）显示 flag 握手不满足——疑与 hash 层补丁的 eid 生成路径或 compressor 拆半近似导致 attention 数值异常（NaN 掩码下的死等）相关；
- 单卡 CPU 专家 offload 模式：输出首个 token 后 **cpumoe C++ 扩展段错误**（core dump，未深入调试）。

### 4.3 对照：dsv41 运行时（V4.1 官方口径）在 A100 上的公开性能

YJH008 仓库实测（V4.1-Flash，同硬件级别）：单流 **38–50 tok/s**、4 并发 ~150 tok/s 聚合。这是 V4-Flash 优化路径的**合理目标带**（V4-Flash 略小：43 层/256 专家 vs 40 层/384，同量级）。

## 5. 我们的引擎 vs 理论上限：差距分解

```
现状 1.7 tok/s (588ms/tok)
  = 43 层 × 895ms/步里每层 ~21ms
  ≈ 76% eager launch 长尾 + 18% 跨卡搬运 + 6% 计算
目标 (dsv41 口径) 45 tok/s (22ms/tok, 27×提速)
```

| 优化项 | 预期 | 依据 |
|---|---|---|
| 1. KV cache + 增量 decode（不再全序列重算） | 必需的地基 | 当前每步重算整序列；v2 引擎已有 PagedAttention 全套可移植 |
| 2. MoE 批处理 gemm（按专家分组一次 gemm，去 python 循环） | op 数 ÷ ~40 | v9 Qwen3MoE 的 GroupedExperts 已实现同款（offsets 一次 tolist） |
| 3. CUDA graph 捕获 decode 步 | launch 开销 → ~0 | vLLM 与我们 3.3x 差距的已知主因（PERFORMANCE_NOTES） |
| 4. MXFP4 原生存储 + in-kernel 反量化 | 权重显存 1.2TB→178GB、带宽 ÷4 | v9.2 dequant-GEMM kernel 可直接复用（DSV4 与 Qwen 同为双 nibble 布局）；也解锁"单卡放不下→8 卡"的显存账 |
| 5. EP8 专家切分 + 分层停留 | MoE 逐卡 32 专家 | dsv41 配方验证过；我们的组件栈（v9.3 GroupedExperts 内置 EP）即为此设计 |
| 6. 跨卡搬运 → P2P 异步/流水 | 160ms/步 → 重叠隐藏 | dsv41 的 `p2p_copy/multicast` 是成熟参照 |

按序投入：1+2 是数量级（~10×），3 再 ~3×，4+5 解锁 8 卡显存与带宽，合起来才是逼近 45 tok/s 的路径。**先做 1**——没有增量 decode，其他都无从谈起。

## 6. 遗留 TODO（按优先级）

1. **KV cache 增量 decode for DSV4**（v2 的 paged attention 移植到 v6.3 的 DSV4 模型；compressor/indexer 的 cache 语义已在运行时里读懂）；
2. dsv41 运行时 decode 死等根因（EP flag 与 hash 层 eid 的交互；compressor Ca|Cb 拆半需按 HF 参考改为正确 overlap 数学——v6.3 CSACompressor 有完整实现可移植）；
3. cpumoe 段错误（单卡 offload 路径）；
4. vLLM 侧 mHC broadcast 的 Ampere fallback（我们 patch 的 `use_tilelang_fallback=True` 方向）值得给上游提 issue。

## 附录：复现资产

| 项 | 路径 |
|---|---|
| dsv41 运行时（含 9 补丁） | `/tmp/deepseekv4.1-A100-custom-main/`（上游 shi3z/deepseekv4.1-A100-custom + 本报告补丁；应固化进仓库） |
| 可写权重副本（patched config） | `/data/ifly_jyhe23/dsv4_rt/DeepSeek-V4-Flash/` |
| 运行时 venv | `/data/ifly_jyhe23/dsv4_rt/venv`（torch 2.6.0+cu124, triton 3.2.0） |
| 我们的 bench+profile 脚本 | `/tmp/dsv4_ours_bench.py` |
| profile trace | `/tmp/ours_dsv4_trace.json`（179MB, 77.7 万事件） |
| vLLM 尝试日志 | `/tmp/vllm_dsv4_1..7.log` |
| dsv41 smoke/诊断日志 | `/tmp/dsv41_smoke1..11.log`, `/tmp/dsv41_diag*.log`, `/tmp/dsv41_single*.log` |
| 56 机器同步的权重副本 | `/data/ifly_jyhe23/dsv4_weights/`（149G，143 原件保留） |
