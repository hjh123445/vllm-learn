# v9 设计：并行 × 量化——同一堵墙的两个攻面

> 状态：设计定稿（实施未开始）。先行实验数据已实测（§II.1.5, §II.2 的 B1 陷阱）。
> v9 = v8 全部 + 量化组件（版本叠加，不独立）。

---

## 0. 项目定位：为什么并行和量化要放一个版本里

推理引擎的根本约束是**显存墙/带宽墙**：每生成一个 token，要把激活参数从 HBM 读一遍、
把整个 KV cache 读一遍。并行和量化是对这堵墙的两个正交攻击面：

```
           每卡读多少字节          每个字节多大
并行策略   TP/EP 切权重, DCP 切 KV   ——
量化       ——                       W4/W8 切权重, FP8 切 KV
```

- **TP/EP**：把权重分到 N 卡，每卡读 1/N——代价是每层通信；
- **DCP**：把 KV 分到 N 卡——代价是 partial/LSE 合并通信；
- **W4 量化**：权重字节 ÷ 3.9——代价是反量化计算（可融合进 GEMM，几乎白送）；
- **FP8 KV**：KV 字节 ÷ 2——代价是 attention kernel 里的 in-kernel 反量化。

**v9 的核心论点**：这两类手段是联合决策而非串联叠加——W4 让 Qwen3-30B 的
60GB 权重缩到 ~16GB，"必须 4 卡装权重"的显存矛盾消失，TP 的存在理由从
"装得下"退化为"分摊带宽"——**量化直接改写并行选型**。决策变量从
(tp, ep, pp, cp) 扩展为 (tp, ep, pp, cp, w_bits, kv_bits)。

---

# Part I：并行策略优化深化

## I.1 资源模型（统一成本框架）

decode 一个 step 的时间下界（每卡）：

```
T_step ≥ max(
    W_traffic / HBM_BW,          # 权重读取: A_active × bytes/param
    KV_traffic / HBM_BW,         # KV 读取: bs × L × 2 × n_layer × kvh × hd × bytes
    FLOPs / FLOPS_peak,          # 计算: 2 × A_active × bs（decode 时几乎不 bound）
)
T_step += T_comm + T_launch      # 通信串行部分 + kernel 启动开销（我们的现实瓶颈）
```

Qwen3-30B-A3B 的具体数字（H20，HBM ≈ 4TB/s）：

| 项 | bf16 | W4 | W4 相对 |
|---|---|---|---|
| 权重/step | 3B × 2B = 6 GB → 1.5ms | 3B × 0.52B ≈ 1.5GB → 0.4ms | **÷3.9** |
| KV/token（48层×4头×128维×2） | 98 KB/token | FP8: 49 KB | **÷2** |

**但注意我们栈的现实**：实测 decode step ~76ms（bs≈3，eager）——比带宽下界高
50 倍，当前瓶颈是 **launch 开销**（28 层 × 每层十几个 kernel 的启动 + python），
这正是与 vLLM 的 1800 vs 5551 tok/s 差距（CUDA graph）的来源。**含义**：
量化的速度收益在 eager 栈上会被 launch 开销稀释（e2e 提速 ≈ kernel 提速的
一个分数），显存收益则立即可得。诚实的度量方案见 §II.3.6。

## I.2 已有实测数据的再解读（v6/v8 累积）

| 实验 | 结果 | 解读（资源模型视角） |
|---|---|---|
| 单机 8 卡矩阵 | EP2 11.3 > TP2 10.7 > TP4 > TP8 8.3 tok/s | MoE 的 EP 切的是 80% 的专家权重，a2c 通信量 ∝ token×topk；TP 切全部权重但 allreduce ∝ bs×H×每层2次，8 卡时环上延迟×跳数吃掉收益 |
| 跨节点（IB 4.5GB/s 实测） | EP2 只损 21%，TP8 崩塌 69-88% | IB 带宽是 NVLink 的 ~1/100：TP 的 allreduce 每层都要过 IB（数据量固定→带宽 bound 崩）；EP 的 a2a 在大 batch 下 token 摊薄 |
| v8.2 DCP | KV 池 575→288 blocks/rank，输出 78/78 一致 | DCP 切的是 KV_traffic——长序列时 KV 读取超过权重读取成为主瓶颈，DCP 的 gather 通信（KB 级）相对 KV 流量（GB 级）可忽略 |
| v7.2 投机 | 5-token verify ≈ 1-token decode（79 vs 76ms） | 再次确认 launch-bound：多 4 个 token 的算力和带宽是白送的 |

## I.3 优化方向清单（按 ROI 排序）

| 优先级 | 方向 | 预期收益 | 依据 |
|---|---|---|---|
| **P0** | CUDA graph 捕获 decode 路径 | ~3x（1800→5000+） | 与 vLLM 差距的已知主因；所有 kernel 级优化的前提（否则收益被 launch 税稀释） |
| **P1** | one-shot custom allreduce（小消息） | decode TP 通信 -50%+ | NCCL ring 对 <1MB 消息是延迟 bound（每层 2 次 × 28 层）；vLLM 同款手段 |
| **P2** | 通信感知调度（研究点 P1/P2 落地） | MoE 场景自适应 | router 先行→预测 expert 负载直方图→失衡系数 max/mean→批内 EP/本地复制决策 |
| **P3** | DCP 交叉点 L* 测定 | 明确适用边界 | KV 节省 ∝ L vs gather 固定延迟 × 层数；扫 L 找 L*（v8.4） |
| **P4** | PP micro-batch | bubble = (p-1)/(m+p-1) | 大模型跨节点时 PP 是唯一不崩的（通信量 ∝ 层边界激活 × m） |
| **P5** | 组合矩阵全扫 | 数据集 | ParallelCoords 已支持任意 (tp,cp,ep,pp)——8 卡扫 TP×EP、TP×DCP、PP×EP 网格，拟合成本模型 |

## I.4 量化改写并行选型（v9 的联合决策实验）

| 场景 | bf16 决策 | W4 决策 | 变化 |
|---|---|---|---|
| Qwen3-30B 单请求低延迟 | TP4（显存） | **单卡**（权重 16GB + 60GB KV 池） | 通信归零 |
| Qwen3-30B 高吞吐 | EP2×TP2 | **EP2 单层** 或 EP4 | TP 层取消 |
| 128K 长序列 | TP8 + DCP2 | TP4 + DCP2 + FP8KV | 等容量下卡数 ÷2 |

v9.4 的实验：同一负载下 (bf16, TP2) vs (W4, 单卡) vs (W4, TP2) 的吞吐/延迟
矩阵——**验证"量化省卡"这个决策路径**（省卡 = 省通信 = 省钱）。

---

# Part II：量化——原理、落地、工程实践

## II.1 原理

### II.1.1 基本式与粒度阶梯

量化是整数近似的缩放重建：`W ≈ s · round(W/s)`（对称）或 `W ≈ s·(round(W/s) − z)`
（非对称，带零点）。**误差完全由粒度决定**——scale 在多大范围内共享：

```
per-tensor（整矩阵一个 s） → per-channel（每行一个 s） → per-group（行内每 g 个连续元素一个 s）
```

### II.1.2 为什么 weight-only 对 decode 有奇效

decode 的 GEMM 形状是 [bs≤32, K] × [K, N]——算术强度 ≈ bs FLOP/byte，远低于
tensor core 的 ridge point。**decode 是带宽 bound：时间 ≈ 权重字节数 / HBM 带宽**。
所以：权重 ÷ 3.9 字节 ⇒ GEMM 时间 ÷ ~2（int4 读 + scale 读）——而不需要激活
也量化（激活在 decode 时只有 bs×K 个元素，可忽略）。**prefill 相反**：大 batch
是算力 bound，weight-only 无益——那是 W8A8 FP8 计算路径（Path C）的战场。

### II.1.3 异常值问题（为什么 naive 量化会崩）

LLM 权重/激活有系统性异常值（个别通道幅度大 10-100 倍）。一个共享 scale 被
异常值撑大 → 其余元素全部塌到几个整数值。**实测**（Qwen3-30B q_proj，本仓库
`v9/tests/quant_probe.py`）：

| 粒度 | int4 相对 MSE |
|---|---|
| per-tensor | **5.25e-01（52%，废了）** |
| per-channel | 2.41e-02 |
| per-group g=256 | 1.81e-02 |
| per-group g=128（AWQ/GPTQ 标准） | 1.54e-02 |
| per-group g=64 | 1.24e-02 |
| per-group g=32 | 1.01e-02 |
| int8 per-channel | 7.32e-05（≈无损） |

两条工程路线解决异常值（比 RTN 更好但需要校准数据）：
- **AWQ**：激活异常通道 → 给该通道乘缩放 s，同时除进权重列
  （`y = W·(s⊙x) = (W·diag(s))·x`，数学恒等，激活变平 → 量化友好）；
- **GPTQ**：用校准数据的 Hessian 做二阶误差补偿，逐列量化并把误差摊给未量化列。

v9 从 **RTN g=128 起步**（零校准依赖、上表证明 1.5% 权重误差对 3B 激活的
端到端影响可测可控），AWQ 作为 v9.x 扩展。

### II.1.4 格式族谱（我们已有的经验）

| 格式 | 位宽 | scale 组织 | 硬件 | 我们的接触 |
|---|---|---|---|---|
| int4 packed | 4 | per-group fp16 | 无原生，kernel 里 unpack | v9.0 实现 |
| int8 | 8 | per-channel | int8 tensor core | probe 已测 |
| FP8 e4m3 | 8 | per-block（128×128）或 per-tensor | **H20 原生 FP8 GEMM** | v6.3 DeepSeek 装载时反量化 |
| MXFP4 | 4+8/32 | 1×32 块共享 e8m0 指数 | Blackwell 原生 | v6.3 DeepSeek 装载时反量化 |
| FP8 KV | 8 | per-head 或 per-block | kernel 内反量化 | v9.3 |

v6.3 已经写过 `dequant_fp4`（int8 双 nibble + e8m0）/`dequant_fp8`（e4m3 +
128×128 块 scale）——那是 **Path A**（装载时反量化）。v9 把它们变成运行时路径。

## II.2 落地路径：四条，收益与成本明确

```
Path A  装载时反量化:  int4 权重 → 装载时全量转 bf16 → 原引擎
        收益: 零（只省磁盘）; 角色: 正确性基线。v6.3 已做。

Path B1 int4 存储 + forward unpack→bf16 matmul（torch 算子级）
        收益: 显存 3.88x; 代价: 【实测更慢】——见下表。

Path B2 int4 存储 + triton 融合 dequant-GEMM（kernel 级，v9.2 核心）
        收益: 显存 3.88x + 权重带宽 ÷3.9 → decode kernel 提速 ~1.5-1.8x

Path C  W8A8 FP8: 权重和激活都 FP8，直接打 H20 的 FP8 tensor core
        收益: prefill（算力 bound）也提速; 代价: 激活量化策略（异常值）
```

**B1 的性能陷阱实测**（`quant_probe.py` §3，H20，N=K=2048，g=128）：

| 形状 | bf16 matmul | B1 unpack+matmul | 比值 |
|---|---|---|---|
| M=8（decode bs=8） | 0.019ms | 0.101ms | **5.48x 慢** |
| M=4096（prefill） | 0.202ms | 0.289ms | 1.43x 慢 |

原因：B1 的 HBM 流量 = int4 读 + **bf16 中间写 + bf16 再读** > 纯 bf16 读一遍；
decode 形状下还叠加了 unpack 的多个小 kernel 启动。**结论：weight-only 的
速度收益只有一条路——把反量化融进 GEMM kernel（Path B2），在寄存器/共享内存
里完成 unpack→scale→dot，HBM 只读 int4。**这是 v9.2 的核心工程。

## II.3 工程实践（每步的坑都在这）

### II.3.1 离线量化器（v9.0）

```
输入: bf16 safetensors checkpoint
输出: 同结构 safetensors: w → uint8 packed [N, K/2]（双 nibble: 低=偶, 高=奇）
                   + scales fp16 [N, K/g]
```

坑：
- **nibble 打包顺序**必须与 kernel 的 unpack 一致（v6.3 MXFP4 的双 nibble
  经验直接复用）；
- scales 用 fp16 存但 kernel 里升 fp32 算；
- 打包后 K 必须 2|K 且 g|K（不满足时 pad，pad 列乘 0）。

### II.3.2 QuantizedLinear 与并行切分的相容性（容易翻车的细节）

量化粒度方向必须与并行切分方向**正交**：

| 权重 | 形状 | TP 切的维 | group 沿的维 | 相容? |
|---|---|---|---|---|
| w13（gate/up） | [E, 2I, H] | I（中间维） | H（输入维） | ✓ group 不跨切缝 |
| w2 | [E, H, I] | I | I？ | ✗ **切缝切断 group** → scales 错位 |
| o_proj | [H, N] | N（列切） | K（输入维=头拼接） | ✗ 同上 |

解法（两选一）：(a) **切分后再量化**（每 rank 对自己的 shard 独立 RTN——
scale 是 shard 局部的，天然对齐）；(b) 约定 group 对齐切缝。v9 用 (a)：
量化器吃"切分后的 shard"，weight_loader 协议不变（v6 已建立）。
EP 切 E 维（专家维）永远与 group 正交 ✓。

### II.3.3 triton 融合 dequant-GEMM（v9.2）

```
grid = (M tiles, N tiles)
每个 program:
  1. load W tile: uint8 [BN, BK/2]（HBM 只读 int4 + 每 group 一个 fp16 scale）
  2. 寄存器内 unpack: 位运算拆双 nibble → int4 → bf16 → × scale 广播
  3. load x tile [BM, BK]（bf16 原生）
  4. tl.dot(bm, wk, acc fp32)   ← 与 v4 flash kernel 同一套手艺
```

坑：scale 的 group 边界与 BK tile 对齐（BK=64, g=128 → 每 tile 恰 2 组，
边界处理写死）；nibble unpack 的向量化（`>> 4` 与 `& 0xF` 各一个 branch-free
op）；fp32 累加（bf16 累加会掉精度——v4 kernel 的数值纪律沿用）。

### II.3.4 数值裁决体系（三级，量化与对拍的裁决标准不同）

```
L1 张量级:   反量化相对 MSE（表 II.1.3 就是）——量化器的单元测试
L2 层级:     单层 forward（fp32 ref vs 量化 bf16）MSE——v6.3 的方法论平移
L3 端到端:   greedy 分歧率（vs bf16 引擎，同 prompt 同 seed）
             + 小语料 perplexity（标准指标）
```

**关键认知：L3 的裁决不是 exact match**——量化有损，逐 token 一致是过强的
标准（bf16 两个 batch size 都可能分歧）。合格线建议：greedy 分歧率 <5% 且
分歧处语义等价、ppl 上升 <0.5%。TP/EP 的数值裁决（逐位一致）与量化的裁决
（统计一致）是两套标准——文档里必须分开写清楚。

### II.3.5 收益可见性的诚实分析（结合 I.1 的 launch-bound 现实）

```
e2e 提速 = (T_launch + T_gemm_bf16) / (T_launch + T_gemm_w4)
```

当前 eager 栈 T_launch 主导 → e2e 提速被稀释（kernel 提速 1.8x 可能只兑现在
e2e 上 1.2x）。**度量方案**：kernel 微基准（真实比值）+ e2e（实现比值）双轨
报告；CUDA graph（P0）落地后重测兑现率。显存收益不受此影响（立即全额兑现：
60GB→16GB 是账面数字）。

### II.3.6 FP8 KV cache 与 DCP 的组合（v9.3）

cache 池 dtype 改 fp8 + 每 (head, block) 一个 scale；v4 的 paged attention
kernel 加 dtype 分支（load 后 in-kernel 升 bf16/fp32）。与 v8 DCP 组合：
**每 rank cache = 1/(cp_size × 2)**。坑：K/V 的幅值分布跨 head 差异大 →
per-head scale 必须；LSE 合并数学不变（v8.0 的 m/l 就是干这个的）。

## II.4 里程碑（每版完整可运行 + 实测裁决）

| 小版本 | 主题 | 裁决标准 |
|---|---|---|
| v9.0 | RTN 量化器 + 误差分析（g 扫描） | **已先行完成**（probe 数据在上表） |
| v9.1 | QuantizedLinear + Path B1 接引擎 | 权重显存 -75% 实测；输出 = 量化 L3 裁决 |
| v9.2 | triton 融合 dequant-GEMM | kernel vs torch 对拍数值一致；decode 吞吐 vs bf16 ≥1.3x |
| v9.3 | FP8 KV cache（+ DCP 组合） | 容量 ×2 实测；长序列输出 L3 裁决 |
| v9.4 | W8A8 FP8 + 联合决策矩阵 + vs vLLM | prefill 提速实测；(bf16,TP2) vs (W4,单卡) 矩阵；与 vLLM awq/fp8 路径对比 |

---

## 附：v9.0 先行实验的完整数据

见 `v9/tests/quant_probe.py`（本文 §II.1.3 / §II.2 的数据来源，2026-10-30
H20 实测，Qwen3-Omni-30B 真实权重 layer.10）。
