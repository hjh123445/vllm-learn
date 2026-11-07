# CUDA Graph batch 补齐：根因、修复与复现

> 2026-11-08 最终更新：本文保留历史排查过程。最新修正、768 组真实图像特征对拍、256 题 Graph/eager 检查及 128/256 并发在线/离线总表见 [最终验证报告](CUDA_Graph多模态最终验证报告.md) 和 [原理说明](CUDA_Graph与多模态原理.md)。

2026-11-07。真实 Qwen3-Omni-30B-A3B-Instruct Thinker 文本主干，172.28.30.56，8×A800-SXM4-80GB，TP8/EP8、DP1、PP1、BF16。PyTorch 2.14.1+cu130、CUDA 13.0、NCCL 2.30.7、Triton 3.8.0、Transformers 5.19.0、driver 580.65.06。

## 1. 结论与验收口径

batch 补齐是正常 CUDA Graph 技术，不能通过只允许精确 batch 绕开差异。最终增加 `--batch-invariant` 策略：在本次已测范围内，未补齐 eager、补齐 eager、Graph replay 的完整 logits 逐位一致，同时检查各 TP rank 一致及有效 KV 不受 capture 污染。严格测试要求 `unpadded_max_abs == 0`，没有放宽容差。

Graph 请求现在默认自动启用严格策略（`VLLM_LEARN_DECODE_GRAPH_STRICT=1`）：固定 GEMM、RMSNorm 和归约顺序，承担性能、通信和显存成本，并且 warmup、KV 预算和 capture 使用同一数值路径。只有显式设置 `VLLM_LEARN_DECODE_GRAPH_STRICT=0` 才会运行形状敏感的快速消融；该模式不承诺未补齐 batch 与 Graph 逐位一致。不能由此宣布任意模型、batch、硬件都逐位一致，也不能将内部一致性当作 HF 整模质量认证。

## 2. vLLM 为什么能补齐

dispatcher 将实际 token 数映射到捕获 bucket，例如 7→8；worker 同步处理 attention metadata，将无效请求长度清零、padding slot 标为 -1，避免假 token 写入真实 KV。Graph 消费固定地址缓冲中的本轮数据。有效请求在数学上独立，但浮点库可能随 M 或通信消息大小改变累加方式。

官方另提供 `VLLM_BATCH_INVARIANT=1`，固定 matmul、归约等数值行为；默认模式不承诺不同 batch 形状生成逐位一致。不能说 vLLM 不补齐，也不能说 Graph 天然保证 batch invariance。

本次核对官方 commit：`1ecc9cc4ce6626352df39e4364579ec376c10761`。

- [官方 batch invariance 文档](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/docs/features/batch_invariance.md)
- [固定 matmul、rank reduction 实现](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/vllm/model_executor/determinism/batch_invariant.py)
- [默认可复现性边界](https://github.com/vllm-project/vllm/blob/1ecc9cc4ce6626352df39e4364579ec376c10761/docs/usage/reproducibility.md)

本引擎 dummy 使用 slot=-1，只读调度器无法分配的专用零 KV block，只返回真实行 logits。与 vLLM 无效长度清零的具体实现不同，都是为了隔离副作用。

## 3. 控制变量与第一处分歧

`v9/tools/diagnose_graph_padding.py` 独立安装 hooks，生产模型不加入拷贝、保存或 profiler。固定权重、前缀、KV 快照和 backend，依次比较未补齐 eager、补齐 eager、dummy token 改值后的补齐 eager、未补齐 eager 重复、补齐 Graph。

从 embedding 开始，比较每层 q/k/v、paged attention、o_proj、router、MoE，找第一处差异，再将 RowParallelLinear 拆成本地 GEMM 与跨 rank reduction 独立验证。专家内部细分也只在工具中包装函数。

### 3.1 batch 3→4：BF16 TP 归约

证据目录：`outputs/mm_graph_fix_20261107/padding_bf16_3to4_final/`，早期版本为 `padding_collective/`。

- embedding、layer0 q/k/v、paged attention 相同，layer0.o_proj 首次最大差 0.001953125。
- 独立本地 GEMM 差 0；BF16 all-reduce 从 3 行变 4 行后差 0.001953125；独立 FP32 all-reduce 差 0。
- 原 BF16 路径最终 logits 最大差 1.625，第一行 argmax 37→1。
- 补齐 Graph 与补齐 eager 差 0；dummy token 0→17 不影响有效结果；非真实新 token slot 的 KV 不变。

证据说明当前 BF16 集合归约对消息形状敏感，不说明 NCCL 算错。未进一步固定 NCCL algorithm/protocol 验证具体内部机制，不能将分块推测当已证事实。最初直接归因 GEMM 不充分，被该组实验纠正。

### 3.2 FP32 all-reduce 仅解决一例，7→8 暴露 GEMM

`padding_fp32_tp/` 的 3→4 整模一致，但 `padding_fp32_7to8_final/` 不成立。layer0.o_proj 首差 6.103515625e-05，仅 4 元素不同；独立 BF16 本地 GEMM 差 1.52587890625e-05，FP32 GEMM 差 0；整模最终 logits 差 5.140625，这组 argmax 恰好相同。

所以自动换 FP32 all-reduce 不是通用修复。代码保留显式 `VLLM_LEARN_TP_REDUCTION_DTYPE=float32` 供消融，Graph 不再偷偷改变默认 TP 数值策略。

### 3.3 固定 GEMM 和 rank reduction 后，发现 RMSNorm

第一版严格路径将首差推迟到 layer10.MoE，最终差 0.09375。单独固定本地专家 sum(0) 后仍差 0.09375，不能说专家求和已被证明是该例根因。

对专家 GEMM 输入再加工具级观测：layer10 attention 输出相同，但 post-attention RMSNorm 后 MoE 输入有 6 元素不同，最大差 6.103515625e-05。该层 router 输出碰巧相同，不能据此推断所有输入相同。

将 RMSNorm 切到每行固定归约的 Triton 实现后，7→8 所有层无差异；随后 batch 1–8、随机前缀、连续 decode、请求重排和混合长度严格验证通过。`padding_invariant_final/padding_diagnosis.json` 中 first_divergence=null。

MoE 离散 top-k 可以放大前序扰动。router 索引变化要结合 same_expert_sets 区分顺序变化与集合变化；专家 ID 的最大绝对差没有数值误差意义。

## 4. 实现位置与代价

`v9/layers/batch_invariant.py` 集中可选数值原语，不全局劫持 aten。

- GEMM 固定 BM=16、BN=64、BK=32 和 K 累加顺序，不按 token 数 autotune，不做 split-K 原子累加；支持线性层与三维本地专家矩阵乘。
- TP/EP 使用 NCCL all-gather 保留每个 rank 的贡献，再按 rank 0→N-1 固定顺序 FP32 累加。数据留在 GPU，通信仍使用 NCCL。
- dense 本地专家结果固定按专家顺序累加。
- `layers/kernels.py` 严格策略固定 RMSNorm、SiLU 路径。
- `linear.py`、`embedding.py`、`moe.py` 调用数值原语；观测留在工具层。
- `_pick_decode_graph` 正常选最小可容纳 bucket，不要求实际 batch 等于 bucket。

all-gather 临时缓冲约 world_size×payload，通信量与 kernel 性能可能劣于 NCCL all-reduce/cuBLAS。固定 GEMM 是可读基线，尚未调优，不能拿旧吞吐数据证明新严格模式更快。

## 5. 验证记录

目录相对于 `outputs/mm_graph_fix_20261107/`。

|目录|内容|结果|
|---|---|---|
|graph_batch_invariant_v1 / v2|前两轮严格测试失败进度|保留，不计通过|
|graph_batch_invariant_v3|重复 token，batch 1–8，17/255/256 长度|三种路径 logits 全相同|
|graph_batch_invariant_random|seed293 非重复随机前缀，batch 1–8|全相同|
|graph_batch_invariant_multistep|seed729，每 batch 连续4步，每步反转请求|全相同|
|graph_batch_invariant_final|seed983，17/255/256/511混合前缀，每 batch4步并重排|全相同，含源码 SHA256|
|graph_batch_invariant_kv_checked|相同输入，三条路径分别恢复独立KV快照，逐步比较全部KV，并同一步反转请求再还原输出顺序|logits、KV与重排输出全部逐位相同|
|padding_invariant_final|逐层、dummy改值、KV隔离|无首差、dummy不干扰|
|graph_invariant_cli_profile|EngineCore两轮三请求生成，八卡trace|真实replay，生成重复一致，审计通过|
|cpu_final.log|仓库回归|144 passed、10 skipped、12 subtests passed|

最终一组有 8 batch×4步×8 rank=256 个 rank-step 比较，检查完整 vocab logits，要求 max_abs=0、unpadded_max_abs=0、tp_ranks_equal=true。合成 token 用于复现，不能当自然语言质量数据。

## 6. 复现

在服务器隔离目录 `/data/ifly_jyhe23/vllm-learn-mm-graph-fix-20261107/v9`，使用 `/home/ifly_jyhe23/env/miniconda3/envs/sglang_contrib/bin/python`。模型路径 `/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct`。以下 python 替换成该环境解释器，MODEL 替换成模型绝对路径。

```bash
# 任意差异失败退出，不放宽容差
python tools/validate_decode_graph.py --model MODEL --tp 8 --batch-invariant --strict-batch-invariant --random-inputs --seed 983 --steps 4 --permute --mixed-lengths --profile --output /tmp/graph_strict_check

# 原始 BF16 3→4 差异和首处分歧
python tools/diagnose_graph_padding.py --model MODEL --batch 3 --bucket 4 --output /tmp/padding_original

# 修复后的逐层对拍
VLLM_LEARN_BATCH_INVARIANT=1 python tools/diagnose_graph_padding.py --model MODEL --batch 7 --bucket 8 --output /tmp/padding_fixed
```

正常 benchmark 使用 `tools/benchmark.py --decode-graph`（该选项自动设置严格数值策略），其余 model/TP/EP/inputs/budget/max-num-seqs 按正常负载设置。加 --profile 只采轨迹，profile 时间不进入吞吐提升表。捕获完成后执行元数据会记录 `graph_capture_count` 和实际 `graph_capture_bytes`；若图池超过 `decode_graph_reserve_bytes`，启动直接失败并提示调大预留或缩小 bucket，禁止静默回退到 eager。

`graph_invariant_cli_profile/trace/cuda_rankN.json.gz` 含单rank CPU/CUDA/NCCL；八卡合并按 PROFILE.md 使用 merge 工具。Graph replay 不执行每层 Python hooks，不能说图内每个节点已有完整 layer 归属；审计显式标记 layer_attribution_verified=false。

## 7. 范围

验证为文本 Thinker、DP1/PP1/TP8/EP8、dense static decode、非量化 BF16、Triton attention/QK。动态 all-to-allv 未进入完整 Graph。图片、PP/DP Graph、量化、多机、自然语言/HF 整模质量不能由这些结果推导通过。严格策略应与相同策略 eager 比较，不要求与旧 cuBLAS/NCCL 策略逐位相同。
