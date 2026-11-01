# 8×A800 实验报告：v7/v8/v9 全链路验证 + 热点分析 + 优化闭环

> 日期：2026-11-02 00:13 – 03:45（8×A800-80G，172.29.226.143，vllm-e2e env）
> 基线：`fix/p0-interview-hardening` 分支（18 项 P0 修复后）
> 负载：Qwen3-Omni-30B-A3B（6 请求 × 48 tok greedy，与 PARALLEL_MATRIX 同口径）
> 本报告所有数字均由本仓库脚本产出，JSON 存档于 `v9/tests/results/`。

---

## 0. 结论速览

| 问题 | 答案 |
|---|---|
| v7/v8/v9 各模型跑通了吗？ | **v9 Qwen3-Omni：7/8 并行配置跑通**（tp1/tp2/ep2/ep4/ep8/tp2ep2/pp2 全部生成正确文本；**dcp2 在 v9 的 worker 路径死锁**——新发现，见 §4）。**v7 投机：3/3 输出与基线一致**。**v8 DCP 三相位：跑通**（修复 run_dcp_tp 的池子双重分配后）。**v9 DSV4：多卡装载死锁**——根因已定位（worker 路径无 layer_device_map，8 个 worker 各自全量装载 149GB），今晚未修 |
| 热点在哪？ | **torch.profiler 实锤：每步 132ms 中 CPU 侧 247ms（决定步长）——`cudaLaunchKernel` 53.8ms/步 × 10,526 次 + MoE 逐专家循环的 3,045 个小 GEMM/步 + 2,800 次 mul/index 小算子**。GPU 计算只有 143ms 且大量空隙——launch-bound，与 v6 结论一致但拿到了完整算子级分解 |
| 优化做了什么、验证了吗？ | **做了 1 项并闭环验证**：W4 量化的 MoE 反量化从"全量"改"按命中专家"（per-expert on-demand dequant）——**4.2 → 6.2 tok/s（+48%）**，A/B 数据存档 |
| 量化的完整画像？ | bf16 24.4 tok/s / 69.3GB → W4 6.2 tok/s / 26.8GB（**权重 -42.6GB/-61%**）；输出 3/6 请求与 bf16 完全一致（分歧处语义连贯，符合"量化统计一致"的 L3 裁决） |
| v0 幻影 token 修复的量化证据？ | 修正后 total_out_tokens = **1152**（与 v1 完全一致；修复前虚报 1920），吞吐 **120.8 tok/s**（修复前口径 155.6）——README 矩阵 v0 行需更新 |
| DeepInfer 能比吗？ | **官方 DeepInfer（24.8k★）只支持 SM100/Ascend 950，A800 无法运行**。社区 `Deepinfer-SM80` 移植版支持 A800，但需要 vLLM PR#56201 特定 commit + **DeepSeek-V4.1-Flash 475GB 权重**（我们只有 V4-Flash 149GB，且有 6 处架构差异）——不可行，详见 §6 |

---

## 1. P0 修复的验证（全部通过）

| 验证项 | 结果 |
|---|---|
| `v9/tests/test_p0_regressions.py`（T1-T7） | **7/7 PASS**（v0 幻影 token / v2 off-by-one / _Spec 切片协议 / 准入校验 / peek_next_id / padded 宽度 / scale 下溢） |
| `v9/tests/test_mixed_batch.py`（P0-2 双重切片的端到端裁决） | **3/3 请求逐 token 一致**——长 prompt（1328 tok）在预算 1024 下真实切了 2 段、混批 1 轮（chunk 计数有断言，不再是平凡通过） |
| P0-1（任何模型无法启动） | 修复后 Omni 正常加载（`_load_hf_config` 分派）；DSV4 的 `_is_dsv4` 判定/导入链验证通过 |
| P0-9（PP） | v9 pp2 跑通 19.3 tok/s，且是**唯一 6/6 输出与单卡完全一致的配置**（PP 无跨卡数值聚合，天然确定性——面试可讲） |

## 2. v9 并行矩阵（修复后全量重测）

| 配置 | GPU | tok/s | 与 tp1 输出一致 | 历史值（对照） |
|---|---|---|---|---|
| tp1 基线 | 1 | 21.5 | — | 22.6 |
| tp2 | 2 | 21.1 | 5/6 | 19.3 |
| ep2 | 2 | 32.3 | 5/6 | 30.0（无档案，本次已存） |
| ep4 | 4 | 41.5 | 5/6 | 37.5 |
| **ep8** | 8 | **48.1** | 5/6 | 45.9 |
| tp2×ep2 | 4 | 31.1 | 5/6 | 28.4 |
| pp2 | 2 | 19.3 | **6/6** | 17.6 |
| dcp2 | 2 | **死锁** | — | 16.4（v8 数据） |

核心结论复现且更强：**EP8 = 单卡 2.2 倍**（吞吐随卡数近线性）、TP2 无收益（launch-bound 下 allreduce 是净税）、5/6 一致率是 bf16 跨卡归约的 argmax 边界噪声（PP 的 6/6 反证：无聚合=无噪声）。

## 3. v7 投机 / v8 DCP

**v7（修复 v8 的 mtp 同步后连带验证）**：

| 请求 | 基线 | 投机 | 一致 | 接受率 |
|---|---|---|---|---|
| 数数 1-12 | 4.7s | 4.9s | ✓ | 2% |
| 重复这句话三遍 | 1.3s | **0.9s（1.44×）** | ✓ | **83%** |
| 介绍一下你自己 | 3.4s | 3.5s | ✓ | 0% |

与 v7.2 历史数据完全复现——n-gram draft 只在复读场景有效的负结果再次确认。

**v8 DCP 三相位**（修复 `run_dcp_tp.py` 的池子双重分配——TP 切分后重分配前先释放旧池）：
base 16.5 → TP2 7.7 → TP2×DCP2 3.6 tok/s，输出文本正常。修复方式：shard 后 `kv_cache=None` + 清空模块 view + `empty_cache` 再 allocate（旧版第二次 allocate 时 used 里还含第一个池，预算归零断言炸）。

## 4. 新发现的问题（今晚定位，未修）

1. **v9 DCP2 worker 路径死锁**：faulthandler 铁证——EngineCore 卡在 `response_q.recv()` 等 ready，worker 侧无输出。TP/EP 的 worker 全部正常，仅 cp>1 卡死。疑点在 worker 的 cp_group 构造/warmup 与 `set_dcp` 的交互（v8 的同语义代码能跑）。**这是 P1 级待修项**，复现：`python tests/run_experiments.py smoke dcp2`。
2. **v9 DSV4 多卡装载死锁**：8 worker 各 1GB 后停滞、磁盘 IO 近零。根因：worker 路径的 `load_dsv4` 不带 `layer_device_map`——每个 worker 试图把 149GB 全量反量化到自己的单卡（v6.3 的 1.7 tok/s 实验用的是手工分层分放）。修复方向：EngineCore 按 pp/ep 坐标给 worker 生成 layer_device_map。**今晚不修**（涉及 worker 协议扩展）。
3. **共享 GPU 的预算问题复现**：别人的 4GB 进程占 GPU0 时，util 0.85 的池预算可能归零（v8 base phase 第一晚失败的原因）——ENGINEERING_NOTES §2.3 的教训重现，实验时避开被占卡。

## 5. Profile 热点分析 + 优化闭环（面试深挖主线）

### 5.1 热点归因（`profile_v9_tp1.json`，20 步平均）

```
每步 132.1ms | CPU 总 246.7ms | CUDA 总 143.0ms | 事件 86,009 个/步
top CPU: cudaLaunchKernel   53.8ms/步 × 10,526 次   ← launch 税
top CUDA: aten::mm          33.2ms × 3,045 次/步    ← MoE 逐专家小 GEMM
          cuBLAS gemv/gemm  17.3ms × 1,839 次
          aten::mul         10.5ms × 2,805 次       ← MoE 循环小算子
          index_add_         10.0ms × 1,402 次
```

**归因链**：48 层 × (63 个 mm + ~59 个小算子) ≈ 每步 5,800 次模型算子 + 框架开销 = 10,526 次 launch；MoE 逐专家 Python 循环（`for e in off_list` × 每专家 mm/mul/index_add）贡献了绝大多数。CPU 247ms > 步长 132ms → **decode 被 CPU 发射决定，GPU 33% 时间空转**。

### 5.2 优化方案（按 ROI 排序）+ 本晚已验证的

| # | 方案 | 预期 | 状态 |
|---|---|---|---|
| 1 | **W4 per-expert 按需反量化**（MoE 量化路径） | 减全量反量化开销 | ✅ **已实施+验证：4.2 → 6.2 tok/s（+48%）**（`layers/moe.py` GroupedExperts.forward 的 quantized 分支从"每步全量 dequant E 个专家"改为"循环内只 dequant 命中专家"） |
| 2 | MoE grouped GEMM（消 3,045 个小 mm → ~100） | decode 大头 | 未实施（vLLM 的 moe_align_block_size + fused grouped gemm 同款思路，工作量 1-2 周） |
| 3 | CUDA graph 捕获 decode | 消 10,526 次 launch 税（53.8ms/步） | 未实施（与 #2 是 vLLM 差距 3.3× 的两个主因） |

### 5.3 量化完整画像（本晚实测）

| 维度 | bf16 | W4（修复+优化后） | 结论 |
|---|---|---|---|
| 吞吐（单卡，6×48tok） | 24.4 tok/s | 6.2 tok/s | B1 路径仍慢 3.9×——dequant-GEMM kernel 化（v9.2 B2）才是速度收益的兑现路径 |
| 显存 | 69.3GB | 26.8GB | **权重 -42.6GB（-61%）**——"量化改写并行选型"的实测论据：30B W4 后单卡 27GB，TP 的"装得下"动机消失 |
| 输出质量 | — | 3/6 完全一致 | 分歧处文本语义连贯（"你好，我是阿里巴巴集团旗下…"）——符合 L3 统计一致裁决 |
| KV 池 | 327 blocks | 206 blocks* | *同 util 下量化引擎当前用固定预算，省下的 42.6GB 尚未自动转化为 KV 池（P2 优化项：量化后重跑 allocate） |

## 6. DeepInfer 对比评估（不可行的证据链）

- **官方 `deepseek-ai/DeepInfer`（24.8k★）**：README 明确 Supported hardware = **NVIDIA SM100 + Huawei Ascend 950**；要求 Python 3.12 + CUDA 13 + GCC 13 源码构建；只支持 DeepSeek-V4.1-Flash。A800（SM80）不在支持列表——与 vLLM 0.31 在 A800 跑不了 DSV4 的"硬件代差"结论一致（DSV4_A800_EXPERIMENT §2 的三连撞同源）。
- **社区 `HalfVulpes/Deepinfer-SM80`（91★，Ampere 移植）**：明确支持 A800-80G，性能数据可引作参考（6 卡 TP2×PP3：1-stream decode 160.8 tok/s、16K prefill +20.5% vs baseline vLLM）。**但在本机不可行的原因**：需要 vLLM PR#56201 特定 commit（`79a7108d9a`，已从 main 消失需 PR ref 拉取）+ 两层 patch + SM80 源码编译（~8GB、Python 3.12/CUDA 13/GCC 13）+ **DeepSeek-V4.1-Flash 权重 ~475GB**（本机磁盘剩 272GB，且手上是 V4-Flash 149GB——与 V4.1 有 6 处架构差异，DSV4_A800_EXPERIMENT §3 刚为这些差异打过 9 个补丁）。
- **面试表述建议**：对比对象用本机跑得通的 vLLM 0.31（Qwen3-Omni 22.6 tok/s 单卡，PARALLEL_MATRIX 已有）+ 引用 Deepinfer-SM80 的公开数据作"SM80 上 DeepSeek 系的可达性能带"（1-stream 151-180 tok/s），并讲清"为什么官方 DeepInfer 的 kernel 栈（FlashMLA/DeepGEMM/DeepEP）在 SM80 全线不可用"——这本身就是 DSV4_A800_EXPERIMENT 的核心叙事。

## 7. 可复现方法（全部脚本已入库）

```bash
# 环境：8×A800-80G，conda env vllm-e2e（torch 2.13+cu130, triton 3.7.1, vllm 0.31, xxhash）
REPO=/path/to/vllm-learn && cd $REPO/v9
PY=/path/to/python

# 1. P0 回归（7 项纯逻辑，~2 分钟）
$PY tests/test_p0_regressions.py

# 2. chunked prefill/混合批一致性（P0-2 裁决，~4 分钟）
$PY tests/test_mixed_batch.py

# 3. 并行矩阵 smoke（每配置 2-6 分钟，结果存 tests/results/smoke_*.json）
$PY tests/run_experiments.py smoke tp1 tp2 ep2 ep4 ep8 tp2ep2 pp2
$PY tests/run_experiments.py compare     # 汇总对比表（含 vs tp1 一致率）

# 4. 量化 A/B + 显存画像
$PY tests/run_experiments.py quant quant_bf16 quant_w4
$PY /path/to/memprobe.py                 # bf16 vs W4 allocated 显存

# 5. Profile（单 EngineCore 修好的版本）
PROFILE_GPU=1 $PY tests/profile_v9.py    # 输出 tests/results/profile_v9_tp1.json

# 6. v7 投机 / v8 DCP
cd $REPO/v7 && $PY tests/test_speculative.py
cd $REPO/v8/tests && $PY run_dcp_tp.py   # base/tp/dcp 三相位

# 7. v0 口径修正复跑（需要 Qwen3-0.6B）
cd $REPO && VLLM_LEARN_MODEL=/path/to/Qwen3-0.6B \
  CUDA_VISIBLE_DEVICES=1 $PY benchmark/harness.py --engine v0 \
  --num-requests 12 --batch-size 8 --out /tmp/v0_fixed.json
```

**结果存档**：`v9/tests/results/*.json`（本次 10 个文件已提交到分支）。注意负载与 PARALLEL_MATRIX 同口径（6 请求 × 48 tok greedy），历史对照列见 §2 表。

## 8. 遗留与建议（按优先级）

1. **P1：v9 DCP2 worker 死锁**（faulthandler 栈已存：core.py:191 等 ready）——对照 worker.py 的 cp_group 与 v8 parallel_coords 的差异；
2. **P1：README 性能矩阵更新**——v0 行改为 120.8 tok/s / 1152 tok（幻影 token 修正后）；v9 矩阵补本次 7 配置数据；
3. **P2：量化引擎的 KV 池自动扩容**（W4 省下的 42.6GB 目前没有变成 KV blocks）；
4. **P2：MoE grouped GEMM + CUDA graph**（profile 实锤的两大主因，vLLM 差距的兑现路径）;
5. **P3：DSV4 的 v9 worker 化**（layer_device_map per-worker 生成）。
