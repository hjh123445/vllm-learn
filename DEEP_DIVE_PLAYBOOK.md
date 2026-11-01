# 深挖主线：从"我做了什么"到"我能推演多深"（面试拷打防御体系）

> 定位：本文是**两条深挖主线的完整思考链路**——A800 上的 DeepSeek-V4（问题→物理约束→路线图→
> 上限推导→Hopper 可迁移点）和 Qwen3-Omni 的调优体系（怎么 profile→热点分层→优化闭环→与
> vLLM 的公平对比框架）。所有数字来自本仓库实测（`v9/tests/results/` + EXPERIMENT_REPORT），
> 所有推导可被面试官当场验算。
> 使用方法：每节先给"30 秒答案"（被问到时的第一反应），再给"深挖层"（追问时的推导链），
> 最后给"诚实边界"（不知道什么/没做什么——主动交代比被戳穿好）。

---

# 主线一：DeepSeek-V4-Flash 在 A800 上如何性能最优

## 1.1 30 秒答案

"A800 是 SM80：没有 wgmma/TMA、没有 FP8 tensor core、HBM 2TB/s。DSV4 官方 kernel 栈
（FlashMLA/DeepGEMM/TileKernels）全部绑定 Hopper+，在这台卡上物理不可用。所以最优路径不是
'移植官方 kernel'，而是**算法层照抄 HF 语义 + kernel 层用 SM80 等价物重写 + 体系层做三件事：
增量 KV cache（消全序列重算）、MXFP4 原生路径（消 8 倍权重带宽）、EP8 专家切分（唯一在
8 卡上实测正收益的并行）**。我推演的可达上限是单流 **~150 tok/s**，依据是同架构社区实现
（Deepinfer-SM80 在 GA100 上 151-180 tok/s）+ 我自己的 roofline 分解，当前我们 1.7 tok/s
的差距每一毫秒都能归因。"

## 1.2 差距的完整分解（为什么是 1.7 tok/s = 588ms/token）

DSV4_A800_EXPERIMENT §5 的 8 步 profile（895ms/步，v6.3 路径）：

| 分项 | 耗时 | 占比 | 根因 | 消掉它需要什么 |
|---|---|---|---|---|
| eager launch 长尾 | ~680ms | 76% | 97,157 op/步（43 层 × ~100 gemm + MoE 逐专家 python 循环） | CUDA graph + grouped GEMM |
| 跨卡搬运 | ~160ms | 18% | `streams.to(dev)` 的同步拷贝 ×5.4 万次 | 停止"分层分放"（EP 化后每卡常驻）+ P2P 异步 |
| 计算 | ~55ms | 6% | bf16 反量化后全序列重算 | 增量 cache + MXFP4 原生 |

**关键洞察**：计算只占 6%——说明"把模型算对"早就完成了，1.7 tok/s 是**纯工程问题**。
这决定了优化顺序：先消 76%（graph/grouped），再消 18%（拓扑），最后才轮到 kernel 微调。

## 1.3 路线图（每步的预期收益 + 依据）

| # | 优化 | 预期 | 依据 |
|---|---|---|---|
| 0 | **增量 KV cache**（HF 五元组状态机：compressed_kv/buffer_kv/buffer_gate/entry_count/overlap_kv） | 必需的地基 | 当前每步重算整个序列——O(L) 变 O(1) 是一切的前提；HF 已把语义踩平，照搬设计（WORKLOG §8.2.1） |
| 1 | **MoE grouped GEMM**（消 python 逐专家循环） | op 数 ÷40 | 我们在 Qwen3-Omni 上实测同款循环是 3,045 个小 GEMM/步；v9 的 GroupedExperts（offsets 一次 tolist）已实现同款，移植即可 |
| 2 | **CUDA graph 捕获 decode** | launch 开销 →~0 | 我们自己的对照数据：Qwen3-0.6B 上 vLLM graph vs eager = 3.3× 吞吐（PERFORMANCE_NOTES §9）；DSV4 的 76% 占比只会更赚 |
| 3 | **MXFP4 原生存储 + in-kernel 反量化** | 权重带宽 ÷4（1.2TB→178GB 常驻） | v9.2 的 dequant-GEMM（双 nibble 布局同款）直接复用；这同时解锁"8 卡各放自己专家"的显存账 |
| 4 | **EP8 专家切分 + 每卡常驻** | 消跨卡搬运 18% + 8 卡并行 | Qwen3-Omni 上 EP8 = 单卡 2.2× 实测（21.5→48.1 tok/s）；DSV4 专家占 94% 权重，切专家即切主要矛盾 |
| 5 | triton 化 partial attention / indexer | 视 profile | Deepinfer-SM80 的 `_sparse_attn_decode` 2.08-2.14× 可参照 |

**叠乘估算**：0 是地基；1×2 消 76% 的 launch → ~7×；3×4 消带宽和搬运 → 再 ~2.5×；
1.7 × 7 × 2.5 ≈ **~30 tok/s 是我们自己路径的保守目标**。要进 150 带，还差 5×——
那 5× 在 kernel 质量（Deepinfer-SM80 量级的手写 MLA/MoE kernel），这就是"上限锚点"的用法。

## 1.4 上限推导（150-180 tok/s 从哪来——可当场验算）

单流 decode 的物理下界（每 token）：
```
权重读取: 3B 激活参数 × 0.52B(MXFP4) ≈ 1.6GB → /2TB/s ≈ 0.8ms
KV 读取:  上下文 L×48层×1 KV头×576维×0.5B(FP8) —— L=8K 时 ≈ 0.11GB → 0.06ms
计算:     2×3B×1 FLOP ≈ 6GFLOP → /312TFLOPS(bf16) ≈ 0.02ms
通信:     EP a2a: 1×6×4096×2B ≈ 49KB/层 ×43层 ≈ 2MB → NVLink 300GB/s ≈ 0.007ms
T_min ≈ 0.9ms/token ≈ 1100 tok/s（8 卡摊 EP 后单卡带宽 ÷8 → 单卡下界 ~140 tok/s）
```
**社区实测锚点**：Deepinfer-SM80（同 SM80、V4.1、6 卡 TP2×PP3 + DSpark K=5）单流 151-180
tok/s——达到理论下界的 15-20%（kernel 质量折损）。我们的 ~30 tok/s 目标是它的 1/5，
差距全在 §1.3 的第 5 步。**面试要点**：能现场推这个 roofline，就证明你理解"最优"是
物理量不是口号。

## 1.5 Hopper 官方优化里能迁移到 A800 的点（逐条判定）

| Hopper 优化 | 本质 | SM80 可迁移性 |
|---|---|---|
| DeepGEMM（wgmma/TMA） | 硬件指令级异步 GEMM | ❌ 指令不存在——**思路可迁移**：SM80 等价物是 cp.async + 多级流水（我们 dequant-GEMM 的"未做 TMA/cp.async 是剩余 gap"注释就是这事） |
| FlashMLA FP8 o_proj | FP8 tensor core | ❌ 但**软件 FP8 可行**（激活 fake-quant 到 e4m3 保持数值——dsv41 运行时已这么对拍） |
| TileLang/CuteDSL kernel | DSL 生成 | ⚠️ triton 是 SM80 的现实 DSL——v4 的手写 triton 手艺直接复用 |
| MLA 结构（MQA + sink + partial rope） | 算法 | ✅ **纯算法层，我们已实现**（对拍过） |
| CSA/HCA 压缩 KV | 算法 | ✅ 算法层——HF 参考实现是蓝图（我们缺的是 cache 状态机，见 1.3#0） |
| MTP/DSpark 投机解码 | 算法 | ✅ v7 已有 n-gram 骨架；**launch-bound 阶段投机几乎白送**（我们实测 5-token verify ≈ 1-token decode，79 vs 76ms——DSpark 在 Hopper 上同理） |
| EPLB 专家负载均衡 | 调度 | ✅ 纯调度层——router 先行预测负载直方图，我们 DESIGN P1/P2 已列 |
| DSpark 置信度调度 verify | 采样 | ✅ 算法层可迁移 |

**规律（面试金句）**：Hopper 的优化分三类——**算法类（MLA/压缩KV/MTP/EPLB）全部可迁移、
数值类（FP8）要软件模拟、指令类（wgmma/TMA）物理不可迁移**。A800 的最优 = 算法类全抄 +
数值类软件化 + 指令类用 triton 重写。这就是"迁移判断"的系统方法，不是逐个碰运气。

## 1.6 诚实边界

- 我们的 DSV4 是 v6.3 单发路径（无增量 cache），v9 的 worker 化还卡在 layer_device_map（WORKLOG §3.2）；
- 1.7 tok/s 的 profile 是 v6.3 的，v9 重构后数字会变（方向不变）；
- 150-180 是引用社区数据做的锚点，我们自己没跑过 Deepinfer-SM80（475GB 权重装不下）；
- FP8 KV 我们还没实现（DESIGN v9.3 规划中）。

---

# 主线二：Qwen3-Omni-30B 的调优探究体系

## 2.1 30 秒答案

"调优不是试参数，是**归因驱动的分层下钻**：先纸面 roofline 定'该担心什么'，再端到端
benchmark 定'差多少'，再 profiler 定'差在哪一层'，最后微基准定'两种写法谁快'——每层都有
可证伪的预测。我们对 Qwen3-Omni 的完整链路：smoke 矩阵（7 并行配置）→ profile（10,526
launch/步、3,045 小 GEMM）→ 优化（W4 per-expert 反量化 +48%）→ 重测闭环。与 vLLM 的对比
结论分档：**kernel 层它赢（带宽打满 vs 我们 launch-bound），调度/并行选型层打平（EP8
2.2× 的结论两边一致），架构理解层我们有自己的增量（chunk 无关性验证、PP 全一致的解释）**。"

## 2.2 调优方法论（四层金字塔——每层干什么、怎么防自欺）

```
L1 纸面模型   回答"该担心什么"     decode 算术强度低 → 带宽 bound → 该测权重/KV 读取
L2 端到端     回答"差多少"         TTFT/TPOT/吞吐 vs 参照系（vLLM/nano）
L3 profiler   回答"差在哪"         Self CPU vs Self CUDA 排序、调用次数 × 模型结构算术验证
L4 微基准     回答"哪种写法快"     写代码前测（v2 的 paged gather 微基准消掉最大不确定性）
```
**防自欺三条**（每条都有本仓库的真实案例）：
1. **跨系统对比先对齐口径**——max_num_seqs 的"每步批大小 vs 总并发"语义差让我们 TTFT
   "好到不信"，重测后才发现（PERFORMANCE_NOTES §9）；
2. **"数字反常地好"就是排查信号**——同上；
3. **profiler 换排序键**——nano 的 guard 税是按 Self CPU 排序才浮出来的（平时大家只看
   CUDA 列）。

## 2.3 我们对 Qwen3-Omni 的完整数据链（拿得出手的每一步）

| 步骤 | 动作 | 发现 | 决策 |
|---|---|---|---|
| 纸面 | 30B-A3B：激活 3B、48 层、GQA 4:1、128 专家 top-4 | decode 带宽 bound（bs=6 时算术强度 ~6 FLOP/B） | 该盯权重读取与 MoE 结构 |
| 端到端 | smoke 矩阵 7 配置 | tp2 21.1≈tp1 21.5（TP 无收益）、**ep8 48.1 = 2.2×**、pp2 19.3 且 6/6 全一致 | 并行选型：EP 优先、PP 只为显存；TP 在小 batch 是税 |
| L3 profile | torch.profiler 20 步 | CPU 247ms > CUDA 143ms；launch 53.8ms×10,526；MoE 3,045 小 GEMM | **瓶颈是 CPU 发射不是 GPU**——kernel 微调无用，该上 graph/grouped |
| 归因验证 | 3,045 ≈ 48 层 × 63 个 mm（128 专家 × ~0.5 命中 + 投影） | 调用次数与结构算术吻合 | 排除测量伪影 |
| 优化 | W4 per-expert 按需反量化 | 4.2 → 6.2 tok/s（+48%），显存 69.3→26.8GB（-61%） | 正确性优先的优化也值钱；速度兑现要 kernel 化 |
| 闭环 | 量化 L3 裁决 | 3/6 输出与 bf16 完全一致、分歧处语义连贯 | 量化的判据是统计一致（TP/EP 才是逐位判据）——两套标准分开讲 |

## 2.4 与 vLLM 怎么比、谁强？（分档诚实对比框架）

| 维度 | vLLM 0.31 | 我们 | 判定 |
|---|---|---|---|
| decode kernel | CUDA graph + 带宽打满（0.6B 上 2.8ms TPOT ≈ HBM 上限） | launch-bound（165ms/步@30B 单卡） | **vLLM 胜**——但差距 100% 归因 graph（我们自己的 nano 对照证明：关 graph 的 vLLM eager 与我们同量级） |
| 调度/连续批处理 | v1 单循环 + chunked prefill | 同语义（v9.4e 重写对齐）+ **chunk 无关性验证**（1328tok×1024 预算 3/3 一致） | 平手——我们多一个可验证性证据 |
| 并行选型结论 | EP 优先（EPLB 配套） | **EP8 2.2× 实测 + TP 负收益的机制解释 + PP 全一致的数值解释** | 平手——我们的小矩阵有自己的解释链 |
| 服务层 | 进程隔离 + zmq + 增量 detokenizer | 线程 + queue（已修异常/竞态/泄漏，但简化） | vLLM 胜 |
| 正确性方法论 | 生产级 | fp32 对拍 HF 100%、**逐位(并行)/统计(量化)双标准**、负结果存档 | 各有所长——我们的方法论故事更完整 |

**面试金句**："和 vLLM 比不是比谁快——是**比谁更知道为什么快/慢**。它赢在工程兑现
（graph 打满带宽），我们在归因链路和可验证性上有自己的增量；且两条结论在两边互相印证
（EP 优先、graph 是第一杠杆），这说明我的测量是可信的。"

## 2.5 下一步该测什么（如果面试官问"然后呢"）

1. CUDA graph 对 30B 的兑现率（我们 165ms/步 → 理论 ~143ms GPU 满载——预期 ~4×，与 0.6B
   上的 3.3× 对照）；
2. MoE grouped GEMM（3,045→~100 mm）与 graph 的叠加效应（两者正交：一个减次数、一个减单价）；
3. 大 batch 下的 TP 复测（当前 6 并发太小——TP 的收益公式里 batch 是分子，bs≥32 时重测
   可能翻案，这正是"TP 什么时候选"的完整答案）；
4. W4 + EP2 的组合（P0-14 修复后的裁决点：量化改写并行选型的直接实验）。

---

# 三、面试拷打预演（把两条主线串起来的追问树）

```
Q: DSV4 在 A800 怎么最优？
├─ 追: 为什么官方 kernel 不能用？→ 三连撞 + SM80 缺什么指令（1.1/1.5）
├─ 追: 那 Hopper 的优化哪些能搬？→ 三类判定法：算法✅/数值⚠️/指令❌（1.5）
├─ 追: 上限多少？怎么算的？→ roofline 现场推 + 社区锚点 + 自己路径的差距分解（1.2-1.4）
├─ 追: 你现在 1.7 tok/s 差在哪？→ 76/18/6 分解表，每项对应一个修复（1.2）
└─ 追: 为什么信 Deepinfer-SM80 的数字？→ 同架构 GA100、开源可复现、且我用自己的
       roofline 交叉验证了它的合理性（0.15-0.2 的 kernel 折损率是行业常识）

Q: Qwen3-Omni 怎么调优？
├─ 追: 第一步做什么？→ 纸面 roofline 定担心什么（2.2）
├─ 追: 怎么知道瓶颈在 CPU？→ profiler 的三个证据 + 算术验证（2.3）
├─ 追: 和 vLLM 比呢？→ 分档表 + "比谁更知道为什么"（2.4）
├─ 追: 你的优化值多少？→ W4 +48%/显存 -61%，含正确性裁决（2.3）
└─ 追: 下一步？→ graph 兑现率/grouped 叠加/大 batch TP 翻案/W4×EP（2.5）

Q: （交叉拷打）你说 EP 优先，为什么 vLLM 默认 TP8 部署 30B？
└─ 答: 三个原因——vLLM 有 graph（TP 的 allreduce 延迟被摊薄）、它面向大 batch 服务
        （带宽 bound 时 TP 线性）、EPLB 还不成熟时 TP 是最稳默认。我们的 EP 结论
        限定在"小 batch + eager"的语境，语境切换结论切换——这正是成本模型的意义。
```

# 四、一句话总纲

两条主线其实是同一句话：**"最优"是物理量（roofline 可推）、"差距"是归因问题（profile
可分）、"修复"是优先级问题（ROI 排序）、"可信"是方法论问题（口径对齐 + 负结果存档）**。
面试官拷打的任何一层，都能沿着这条链路下钻到有数据的底。

## 五、图编译优化的裁决实验（2026-11-02 04:00-05:15 补，GPU 实验 07:00 前全部停止）

**问题**：图编译（CUDA graph / torch.compile）考虑了吗？——考虑了、排了最高优先级
（profile: launch 税 53.8ms/步 × 10,526 次），但**还没实现**。本节把"为什么还没"
从判断变成实证：三个假设，先预测后裁决（`tests/graph_probe.py`，结果 JSON 同目录）。

| # | 假设（先写预测） | 裁决（后跑实验） | 面试含义 |
|---|---|---|---|
| H1 | 朴素 CUDA graph 捕获 decode forward 会失败——MoE 的 `offsets.tolist()`（D2H 同步）在捕获期非法 | ✅ **证实**：`RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless...pin_memory()`——整个 run 和仅 model forward 两种范围都在 MoE 处炸 | **graph-safe MoE 是 CUDA graph 的硬前置**（vLLM 的 `moe_align_block_size` 存在的理由），不是"加个开关"的事 |
| H2 | 30B 的 GPU 空转率低于 0.6B 的 87%（kernel 更大） | ✅ **实测 63.4%**（同窗口法：profiler 内 wall 4.367s vs device busy 1.598s）——同时修正了我旧 profile 的方法学错误（旧法用无 profiler 步长减有 profiler CUDA 时间，得出 -11ms 负间隙的荒谬值） | **CUDA graph 的收益上界 = 消掉 63% 空转** ≈ 步长 165→~100ms（若 GPU 满载）；"两个数必须同窗口测"是 benchmark 教训的又一例 |
| H3 | torch.compile 会遇 guard 风暴（每步 Context/张量重建） | ⚠️ **三连发现，比预测更糟**：(a) bf16 下显存就不够（权重 61G+ctx 6.4G，util 0.8 剩 0.6G<激活峰值）；(b) 换 W4 路径后两种模式（reduce-overhead/default）都在 MoE 处**编译期物化 19.66GB 巨型张量**（10.5G 元素）OOM——compile 把逐专家循环向量化时内存放大；(c) 只 compile attention 绕开 MoE，则炸在我们手写 triton kernel 的 **fp32/fp64 循环变量类型漂移**（`m_i` 被 Inductor 拒绝） | compile 路线在本栈的三个前置：graph-safe MoE、kernel 类型纪律（v4 的 `input_precision="ieee"` 教训的延伸）、编译期显存预算 |

**顺手根修**：H3 的 W4 路径暴露了"量化省显存不回流"缺口（WORKLOG §3.3）——
`ModelRunner` 增加 `quantize_int4_w4` 构造参数，**装载后、分池前**在线量化：
KV 池 327 → **1643 blocks**（省余量 41GB 全部回流）。这是"量化改写并行选型"
论点的最后一块实现（单卡 27G 权重 + 41G 池）。

**结论（图编译的正确路线）**：
1. 短期：先修 MoE 的 tolist（offsets 预估/pinned 或 moe_align_block_size 式重写）
   → 才能朴素捕获；attention kernel 的类型纪律顺手修（`m_i` 显式 fp32）；
2. 中期：手写 CUDA graph（vLLM 路线：静态形状 buffer + 捕获/回放 + 多 batch 档位）
   比 torch.compile 可控——compile 在 MoE 上的内存放大是结构性的；
3. 收益锚点：H2 的 63.4% 空转 × 我们的 165ms/步 → 理论 ~100ms（1.65×）；
   叠加 grouped GEMM（消 3045 小 mm）才是与 vLLM 的 3.3× 差距的完整兑现。
