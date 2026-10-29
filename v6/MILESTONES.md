# v6 MILESTONES：小版本工作文档

每个小版本独立成节：目标 → 实现 → 验证 → 踩坑（现象-排查-根因-解决）。
小版本不改变目录结构（都在 v6/ 里增量交付），演进以本文档为准。

---

## v6.0 —— Qwen3-Omni-30B MoE 单卡推理

**目标**：v5 引擎（dense）支持 MoE 模型，fp32 数学对齐。

**实现**：
- `layers/moe.py`：MoERouter（fp32 softmax→topk→normalize）+ GroupedExperts
  （128 专家权重融合 `w13[E,2I,H]`/`w2[E,H,I]`，按专家分组 gemm——非逐 token 复制）；
- `models/qwen3_moe.py`：thinker 文本路径（attention 栈复用 v4 的 triton flash）；
- `utils/loader.py`：thinker 前缀剥离、专家融合装载、跳过 8269 组 vision/audio 权重。

**验证**：
- 端到端输出正确（自我介绍/天空/1+1=2）；
- **MoE 层 fp32 单元对拍 max diff = 0.0**（与 HF router+experts 数学逐位一致）；
- 纯文本 mrope 数学等价性证明（inv_freq diff 1.9e-9，cos 差 0.002 < bf16 分辨率）。

**踩坑**：
1. **einsum 逐 token gather 专家权重 → 768GB 显存爆炸**。根因：`w13[sorted_expert]`
   按 token 复制整份专家矩阵。解决：按专家分组循环（每专家一次 gemm，激活只含
   命中 token）——MoE 实现第一大坑，vLLM 用 fused grouped gemm kernel 从根上避免；
2. bf16 端到端在 router topk 边界分歧（fp32 单元对却深处分歧）——MoE 对精度的
   敏感性比 dense 高一个量级：router logits 噪声翻转专家选择 × 48 层放大。
   双层验证策略（逐位单元 + 统计端到端）的必要性由此而来。

---

## v6.1 —— TP 张量并行（单机 TP2/4/8 + 跨节点 TP8）

**目标**：Megatron 式层内切分：q/kv/o 按头、MoE w13 gate|up 各半 + w2 列切、
每层两处 allreduce 同步。

**实现**：
- `engine/parallel.py`：`shard_model_for_tp`（含 kv 头数 < TP 时的复制语义）、
  `tp_allreduce`、TP 子组管理；
- `tp_worker.py` + `run_tp.py`：独立进程启动器。

**验证**：TP2/TP4/TP8 输出与单卡完全一致（3 请求 greedy 逐 token）；
跨节点 TP8（本机 4 + 节点 56 4）跑通且正确。

**实测**：TP4 单机 10.0 tok/s vs TP8 跨节点 1.2-3.7 tok/s——**卡数翻倍吞吐崩塌
69-88%**。归因：每步 2.4MB 通信在 NVLink（8µs，不可见）vs TCP（25-100ms）×
48 层边界——PARALLEL_ANALYSIS §8 的完整定量。

**踩坑**（三个，全部记录在案）：
1. **⭐⭐ narrow().contiguous() 视图陷阱**：切分后显存 61GB 纹丝不动。
   排查五步：params 32GB vs allocated 61GB 的 29GB 幽灵 → memory_snapshot
   的 0.81GB block（旧 w13 整块尺寸）→ gc 无 Python 引用 → 隔离复现
   （`Parameter(w.data[:500].contiguous())` 不降 / `.clone()` 立降）→ 根因：
   **连续切片的 contiguous() 返回 view 共享 storage**。切分点全部换 clone()。
   同一根因 TP 时踩过一次（绕过），EP 时才逼出——绕过的坑会回来找你；
2. **spawn + mp.Manager 乱码**：同一套代码 spawn 子进程稳定输出乱码，
   换独立 python 进程完全正确。根因未深究（疑似 Manager pickle 与子进程
   CUDA 初始化时序），保守方案：每卡一个独立进程，结果走文件；
3. **kv 头数(4) < TP8**：切不动 → 复制不切（`max(1, kv//tp)`），多个 rank
   持有同一 kv 切片。

---

## v6.2 —— EP 专家并行（单机 + 跨节点）

**目标**：专家对半分（每卡 64/128），DP-style 实现：输入冗余、本地部分和、
每层一次 all_reduce 聚合（v6.4 升级 dispatch 式 all_to_all）。

**实现**：`shard_model_for_ep` + `ep_worker.py`（文件化 worker——ssh 传
多行 python 源的 `\n` 字面化坑与 TP 同解法）。

**验证**：EP2 单机/跨节点输出均与单卡一致；MoE 层 NCCL 单元测试 EP4 diff=0.0。

**实测**：EP2 单机 11.8（全场最优）> TP4 单机 10.0 > EP2 跨节点 9.3（只损 21%）
>> TP8 跨节点 1.2-3.7。两个结构性结论：
- **EP2 双卡 > TP4 四卡**：EP 计算量跟激活参数走（每卡半数专家），
  TP 跟 token 走（每卡碰全部层）；
- **跨节点 EP 只损 21% vs TP 崩塌**：每层一次聚合 vs 每层两次串行 allreduce。

**踩坑**：EP4/EP8 引擎层输出退化（MoE 层单元测试正确）——open issue。
排查过程：采样分叉假设 → 加 rank0 广播（修复死锁，EP8 能跑完）→ 仍空 →
逐层 norm 对比正常 → 候选根因 bf16 聚合精度 × 48 层放大。**修复方向 =
v6.4 调度集中化**（与 PP lockstep 同根）。

---

## v6.3 —— DeepSeek-V4 模型文件（进行中）

**架构调研完成**（权重结构 + HF 源码全读）——远超 MLA+FP8 的复杂度：

| 组件 | 内容 | 状态 |
|---|---|---|
| FP8 block 量化 | int8 存储 + e8m0 scale（128×128 块） | ✓ 反量化对拍通过 |
| 路由 | sqrtsoftplus（√softplus 打分）+ 前 3 层 hash 路由（tid2eid 查表）+ routed_scaling 1.5 | ✓ 单元通过 |
| MLA | q_lora 1024 → 64 头×512 dim；kv 单投影 512 + 压缩 KV（CSA/HCA compressor）| 待做 |
| attention sink | 每头可学习 sink logit（gpt-oss 式）| 待做 |
| hyper-connections | 4 残差流 + Sinkhorn 双随机混合（mHC）| 待做 |
| yarn rope | factor=16, original 65536 | 待做 |
| MTP | num_nextn_predict_layers=1（v7 投机解码直接用）| 待做 |

**第三阶段实测**（tests/test_dsv4_e2e.py）——管线全通：
- `models/deepseek_v4.py`：完整模型（43 层 MLA+MoE+mHC）；
- `utils/dsv4_loader.py`：MXFP4 + FP8 双格式反量化装载（1038 张量）；
- **8 卡分放跑通**（反量化后 ~584GB bf16：FP4 专家×4 + FP8 其他×2，
  单卡放不下是数学必然）——layer_device_map 分层 + 前向跨卡自动搬运激活；
- 生成 16 token 完成，输出目前是乱码（管线正确性 ✅，数值正确性待查——
  已知简化项见下）。

**已知简化项（乱码的候选原因，按嫌疑排序）**：
1. yarn rope 未实现（config: factor=16, original_max=65536）——当前用
   theta=10000 的标准 rope 近似，长位置下频率完全错；
2. HCA/CSA 压缩器未接入（长上下文组件，短序列影响应小但 block_bias 语义
   可能影响注意力分布）；
3. 反旋转的共轭方向可能有正负号问题（HF 是 query 位置的 -sin）。

**第三阶段踩坑（4 个，全部记录）**：
1. **模型构造 fp32 陷阱**：`torch.empty` 默认 fp32 → 搬运时每个 w1 是
   8.6GB（bf16 的两倍），4 层就爆卡。修：搬运带 `dtype=dtype`；
2. **shard 中途搬运垃圾**：第一个专家装完就把含 255 个未装垃圾的整 param
   搬上 GPU。修：统一在装载完成后搬；
3. **反旋转半宽 bug**：cos/sin 是半宽（32 维覆盖 64 维 rope），
   `rd = cos.shape[-1]` 把覆盖区算成了 32——应为 `2 * half`；
4. **跨卡设备的四个散点**：input_ids/embed 设备、rope buffer 共享单卡、
   路由 fp32 产物、einsum dtype 混合——分层分放模型的必然代价。

**第二阶段实测**（tests/test_dsv4_stage2.py ALL PASS，layers/dsv4_components.py）：
- HyperConnection（mHC 4 残差流 + Sinkhorn 20 轮）：**与 HF 逐位对拍 diff=0.0**
  （post/comb/collapsed 三个输出全对）；
- HCA 压缩器（128:1 窗口压缩）：**与 HF 逐位对拍 diff=0.0**（64 tok → 2 条压缩 KV）；
- MLA 核心数学：无 sink 时精确退化为普通 attention（diff=0.0）+ partial rope
  维度语义（前 32 维旋转/后 32 维不变）验证。

**第二阶段踩坑**：HF 组件对拍的输入维序是 [B,S,H]（我们平铺是 [T,H]）——
HCA 对拍时传反了维序导致输出 0 长度；HF 的 rope 初始化依赖完整 config
（yarn 等），对拍用 nn.Module 的 NoRope monkeypatch 跳过。

**第一阶段实测**（tests/test_dsv4_stage1.py ALL PASS）：
- FP8 反量化：w1 [2048,2048] → 均值 -0.0001 / std 0.77（合理分布），块级
  手算自检 diff=0.0；
- sqrtsoftplus 路由：top6 归一化 × scaling 后权重和精确 = 1.5；
- hash 路由：tid2eid [129280, 6] 查表语义验证。

**踩坑（FP8 反量化的三个坑，全记录）**：
1. **e4m3fn 的 NaN pad 字节是 0x7F 和 0xFF（=127 和 -1）**——不是直觉的
   ±127/±128。逐字节扫描 256 个值定位（`torch.arange(-128,128).view(float8_e4m3fn)`）。
   0.36% 的字节是 pad，替换为 0 后分布正常；
2. **e8m0 scale 的正确解码是直接 `.float()`**（torch 已实现 2^(bits-127)
   语义）；`view(torch.uint8)` 拿到的位模式与存储值不一致（120 存储值
   view 出 134）——不能用位操作解码；
3. int8 存储的 e4m3 需要 `view(torch.float8_e4m3fn)` 位重解释后 `.float()`，
   与直接 int8 数值不同（-84 → -0.375）。

---

## v6.4 —— 调度集中化（待做，三个 open issue 的共同解）

现状：各 rank 独立跑 scheduler、假设决策相同——在请求完成边界/抢占边界脆弱，
导致 EP>2 退化、PP lockstep 死锁、TP×EP 重复相加。
方案：**rank0 独占跑 scheduler，决策（batch 组成/块分配）broadcast 到全部 rank**
（vLLM 同款）。这是 v6 剩余工作的第一优先级。
