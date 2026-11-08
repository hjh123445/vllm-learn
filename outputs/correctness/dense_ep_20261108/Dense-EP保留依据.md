# Dense-EP decode 是否值得保留

2026-11-08，基于已发布实现 `3f9f968`（v9 生产代码与 `b3bed86` 相同）。

**结论：保留为默认关闭的可选路径。** 在当前 A800/TP8 自研引擎上，实测 Dense-EP + CUDA Graph 比普通及 packed 稀疏 EP eager 更快；直接删除会失去当前唯一已验证的完整 decode Graph 路径。它用更多计算换掉动态通信、CPU 同步和提交开销，是当前实现的取舍，不代表全专家计算是 MoE 的普遍最佳方案。

## 实现、作用与代价

普通稀疏 EP：router 选择 top-k，将 token 通过变长 all-to-all 分发到专家 owner，仅计算选中的专家，再 combine。当前 Python/PyTorch 路径仍有动态 counts/splits 的主机回读，以及按专家分组的控制/提交开销。严格 batch-invariant 模式还会使用逐专家 loop，原生 grouped API 在本环境也不保证成为一个融合 device kernel。

Dense-EP decode：在 DP1 的 TP 副本内，各 rank 输入相同，独立 router 获得相同路由。每个 rank 用固定形状 BMM 计算自己所有本地专家对全部 token 的输出，未选中的专家贡献权重为零，再作跨 rank 归约。它保留 top-k 对输出的选择，但没有保留稀疏计算量。所有 rank 合计，每个 token 计算 128 个专家，而原模型选 8 个，**专家矩阵计算的理论 FLOPs 约多 16 倍**；整模耗时不按这个倍数变化。

固定形状避免了动态 expert dispatch/counts，使本实现能够捕获整段语言 decode 前向。CUDA Graph 本身并不要求计算全部专家：设备侧动态路由、固定容量 dispatch buffer、支持 device offsets 的稀疏 grouped kernel，或分段捕获，都可以作为后续实现方向；这些替代尚未在本项目完成。

## 本轮直接对照

硬件/模型：单机 8×A800 80GB，Qwen3-Omni-30B-A3B-Instruct、BF16、TP8+EP8、DP1/PP1/CP1。使用真实完整权重，前缀为固定 seed 的 128 个 token，测量位置 129 的完整 decode 步。batch 为 1/8/32/128/256。

测试两组独立运行：默认 unpacked EP，以及显式 `VLLM_LEARN_EP_PACKED=1`。每组对照普通数值策略的稀疏 eager、严格策略的稀疏 eager、严格 Dense-EP eager、严格 Dense-EP Graph。稀疏普通路径显式选择 `torch_grouped` 后端，严格路径按当前策略转为逐专家固定数学 loop；Dense 路径选择 BMM。每项先 warmup 2 次，再测 6 步，共 3 轮，轮换模式顺序，报告单步耗时的中位数。两组合计 **120 项计时记录、720 个被计时完整前向步**，另有不计时的 warmup/prefill/capture。

计时包含 `ModelRunner.run` 的 CPU 输入准备、模型前向、logits、采样和输出 token 回读，使用 CUDA synchronize 定界，取 8 rank 最大 wall time。每种模式从同一份 prefill KV 恢复，输入 token、位置、context 长度一致；同项重复 decode 会覆盖相同的 KV 槽位，不追加新生成 token。

单位为 ms/步，越小越好。“普通”表示未开启 batch-invariant；“严格”表示开启该策略。不同路由/归约路径不承诺逐位等价，表中普通策略只作为另一条实际可用性能基线。

| batch | 稀疏普通 | packed 稀疏普通 | 稀疏严格 | packed 稀疏严格 | Dense eager 严格 | Dense Graph 严格 | packed 组的 Dense Graph | 同组 Graph 对 packed 普通加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 103.29 | 101.36 | 109.09 | 106.19 | 56.96 | 20.01 | 20.08 | 5.05× |
| 8 | 119.56 | 116.60 | 132.79 | 126.41 | 58.74 | 21.16 | 21.25 | 5.49× |
| 32 | 124.44 | 119.62 | 139.61 | 134.92 | 58.87 | 24.65 | 24.69 | 4.85× |
| 128 | 126.49 | 120.98 | 148.22 | 143.56 | 60.64 | 41.13 | 41.21 | 2.94× |
| 256 | 125.87 | 125.83 | 152.85 | 150.29 | 68.15 | 65.21 | 65.26 | 1.93× |

“packed 组的 Dense Graph”不执行 packed dispatch；Dense 路径忽略该开关。重复测它用于给 packed 稀疏提供同一次运行内的 Graph 对照，也保留两次运行的波动，不把跨运行不同分母拼在一起。

## 如何解释这些数字

当前实现存在显著的调度/通信/提交成本，所以即使多算专家，Dense 路径仍可能更快。这个受控实验说明它在当前实现中有实际用途，不能推导出 Dense 比优化后的稀疏 MoE 算法更优。batch 增大时，冗余计算的代价逐渐显现；是否在其它 GPU、更大 batch、不同路由分布或不同上下文下合算，需要重新测量。

**这些是固定输入的完整 decode 步测试，不是自由生成、在线并发吞吐、TTFT 或端到端请求加速倍数。** 权重加载、prefill、图像 encoder、HTTP、队列和 capture 不计入本表；本轮也不比较各模式显存，因为它们共用已分配的 Graph pool。表中的 128-token 前缀不能代表所有长上下文负载。

此前中档请求矩阵的 +10.92%/+2.18%，比较的是 **相同 Dense-EP 和相同数值策略下 Graph 与 eager**，只隔离 Graph 回放收益，不能用来证明 Dense 相比 sparse 的收益。本轮才补上直接 sparse/packed 对照；两套实验的计量窗口不同。

正确性结论仍沿用已发布的严格 logits/KV 回归与 256/256 Graph/eager 生成对照；本性能脚本没有新增 sparse/dense 的逐位等价保证，也没有解决已记录的 HF 250/256 或 DP2×TP2 BF16 差异。

## 保留方式和后续替代条件

- 普通引擎仍默认 `VLLM_LEARN_EP_DENSE_DECODE=0`，走稀疏 EP；没有为本次测试修改生产默认值。
- 需要当前完整 decode Graph 时，显式开启 Dense-EP，满足 DP1/PP1/CP1、非量化专家、Triton attention 等支持条件。
- 当前测试保留该路径；不把 Dense 当作所有 EP 请求的统一后端。
- 后续若实现真正可捕获的稀疏路径，应在同输入/数值契约下通过 logits/KV、生成正确性和性能对照，再据覆盖范围决定替换或删除 Dense 路径。

## 原始证据与复现

生产源码未修改，107 个 v9 Python 文件的原始 SHA256 与上一轮最终快照一致。原始命令、退出状态、每轮数据、源码指纹及日志位于 [本轮证据](../outputs/correctness/dense_ep_20261108/)。脚本为 [benchmark_dense_ep.py](../experiments/dense_ep_20261108/benchmark_dense_ep.py)。

```bash
python experiments/dense_ep_20261108/benchmark_dense_ep.py \
  --source "$PWD" --model /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --output outputs/my_dense_ep_comparison

VLLM_LEARN_EP_PACKED=1 python experiments/dense_ep_20261108/benchmark_dense_ep.py \
  --source "$PWD" --model /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --output outputs/my_packed_ep_comparison
```

脚本为受控消融，在内存中显式切换专家路径与 Graph，保留同一组权重、KV 和捕获图；不改变生产代码，也不修改生产 Graph 的错误处理和支持边界。复现请独占八张卡并用新输出目录。
