# 输出索引：先选一组，再看对应文件

不再保留几十个含义相近的compact/preview/merged/final版本。每组profile只有原始rank记录、一个完整合并文件和一份分析。**“完整合并”指没有裁剪已记录事件，不保证原始采样没有漏尾。**

## 当前工具验证：先打开这个

最新全程采集见 [profile_fullrun_20261106](validation/profile_fullrun_20261106/README.md)：普通 serve/bench 只加 `--profile`，默认首 forward 到关闭；41项主回归通过，另补独立CLI验收回归。八卡各16消息逐条检查 Attention/MoE 的 CPU/GPU 覆盖。先打开 [八卡完整trace](validation/profile_fullrun_20261106/tiny_pp2ep4/trace.json.gz) 或 [概览](validation/profile_fullrun_20261106/tiny_pp2ep4/analysis/timeline.html)。这是自生成小模型工具验证；旧30B缺尾仍标为不完整。

最新精度排查见 [DP2×TP2/EP4 cached generation](correctness/ep_diagnosis_20261106/README.md)：
逐层捕获、32 个层批次逐 route 重放和精度消融。仅专家 FP32 消除合成反例；
默认 BF16 的原严格生成回归仍失败。这里是数值证据，不是 profile 或 30B 性能成绩。

运行时清理后的验证见 [runtime_cleanup_20261106/final](validation/runtime_cleanup_20261106/final/README.md)：已删除lane和PP形状header，八卡PP2×TP4+EP4，保留各卡CPU、ATen、CUDA stream与NCCL。先看 [概览](validation/runtime_cleanup_20261106/final/tiny_pp2ep4/analysis/timeline.html)，再打开 [完整trace](validation/runtime_cleanup_20261106/final/tiny_pp2ep4/trace.json.gz)。这是自生成小模型的功能/采集验证，不是30B性能或质量认证。

下面两组来自同日较早的工具验证快照，先于上述运行时清理；`current`是历史目录名，不代表自动跟随当前源码。

| 目录 | 实际模型与负载 | 并行与轨迹 | 用途 |
|---|---|---|---|
| [tiny_pp2ep4_current](profiles/tiny_pp2ep4_current/) | 自生成8层/hidden16/4专家小模型；8请求、8输入token、4生成token；实际窗口见capture | 八卡PP2×TP4+EP4，每卡CPU、ATen、CUDA、NCCL，stage0层0–3/stage1层4–7 | 较早工具快照，验证全卡采集；不是30B性能 |
| [tiny_ep4_current](profiles/tiny_ep4_current/) | 同一小模型与输入 | 四卡TP4+EP4，单stage | 对照无PP的采集链 |

较早快照的精确命令、依赖与源码在各目录 `manifest.json`；`inputs.json` 是实际输入，`run.log` 是执行日志。

## 真实30B：历史版本，存在尾部采样缺口

共同条件：2026-11-04，同机8×A800，Qwen3-Omni-30B-A3B-Instruct的文本Thinker；8请求，每条47输入token，生成4token；profile预算512，微批预算128，采集从第1条消息开始，上限8条。保存的源工作负载 `inputs.json` 写max_tokens=32，但**实际capture命令覆盖成4**，可由manifest的命令与outputs验证。

| 目录 | 当时拓扑/变化 | 主要轨迹 | 已知缺口 |
|---|---|---|---|
| [qwen_ep8_legacy](profiles/qwen_ep8_legacy/) | 旧独立EP8轴、单stage；不是当前Attention TP8+EP8 | [八卡trace](profiles/qwen_ep8_legacy/trace.json.gz)、[概览](profiles/qwen_ep8_legacy/analysis/timeline.html) | 多rank最后decode消息缺尾层；rank5更早缺尾 |
| [qwen_pp2ep4_legacy](profiles/qwen_pp2ep4_legacy/) | 旧PP2×独立EP4，普通P2P组 | [八卡trace](profiles/qwen_pp2ep4_legacy/trace.json.gz)、[概览](profiles/qwen_pp2ep4_legacy/analysis/timeline.html) | 最后消息尾层不完整 |
| [qwen_pp2ep4_p2p_legacy](profiles/qwen_pp2ep4_p2p_legacy/) | 同旧拓扑，专用P2P communicator | [八卡trace](profiles/qwen_pp2ep4_p2p_legacy/trace.json.gz)、[概览](profiles/qwen_pp2ep4_p2p_legacy/analysis/timeline.html) | 同类尾部缺口；不宣称已证明端到端优化 |

每组均保存 `raw/cuda_rank0..7.json.gz`，不是只有record_function的示意图。每卡约2.6万–3.3万个GPU kernel和大量ATen；合并文件约75–77MiB，展开后可能超过1GB。`raw/pipeline_rank*` 仅CPU调用；不要拿它代替CUDA图。文件内部的旧原路径是采集来源，不要求本机存在同一路径。

`analysis/report.md` 列出基础验收与缺口；`summary.json` 给出精确统计。旧轨迹用于理解历史瓶颈和练习读图，不能认证当前V1/DP×TP架构或拿来报告本轮新吞吐。

## 其他必要证据

| 目录 | 保留什么 | 如何解释 |
|---|---|---|
| [kernels/silu_matched](kernels/silu_matched/) | 同入口PyTorch/Triton eager/Graph计时、5种T的独立CPU/CUDA trace、逐kernel解析 | 单算子后端选择依据，见[性能分析](../docs/性能分析.md) |
| [benchmarks/legacy_20261104](benchmarks/legacy_20261104/) | 当时自研/vLLM同机原始逐请求结果、输入/配置、运行队列与源码部署记录、汇总 | 历史无profile性能，不是当前版本成绩 |
| [correctness/legacy_20261104](correctness/legacy_20261104/) | HF参考token、固定前缀请求、数学消融、batch-shape摘要 | 384/384是12×32固定前缀位置，不是任务准确率；部分大logits未纳入仓库 |
| [correctness/coupled_20261105](correctness/coupled_20261105/) | 新DP×TP拓扑的NCCL小模型日志、命令、源码SHA、选定原始logits包 | 不等于真实30B完整质量认证 |
| [correctness/pp_v1_20261105](correctness/pp_v1_20261105/) | V1队列/PP生产worker的小模型回归证据 | 证明测试覆盖的调度/通信路径 |
| [validation](validation/) | 本次工具回归与无profile小模型CLI验证 | 仅工具与功能验收，不用小模型速率代表30B |

`tests`是会重新执行并判断对错的程序；`outputs`是某次执行留下的数据；`docs`解释方法和边界。三者不能互相替代。重采命令见 [Profile手册](../docs/PROFILE.md)。
