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

## v6.3 —— DeepSeek-V4 模型文件（待做）

目标：`deepseek_v4` 架构——MLA（q_lora_rank 1024 / qk_rope_head_dim 64 压缩
注意力）、256 专家 top-6 + 1 shared、sqrtsoftplus 新路由、FP8 block 量化
（128×128）反量化。149GB @ /data/xmhuang6/deepseek/（只读）。

---

## v6.4 —— 调度集中化（待做，三个 open issue 的共同解）

现状：各 rank 独立跑 scheduler、假设决策相同——在请求完成边界/抢占边界脆弱，
导致 EP>2 退化、PP lockstep 死锁、TP×EP 重复相加。
方案：**rank0 独占跑 scheduler，决策（batch 组成/块分配）broadcast 到全部 rank**
（vLLM 同款）。这是 v6 剩余工作的第一优先级。
