# 稀疏专家计算与并行 CUDA Graph 验证

实现说明见 [稀疏 MoE 与 CUDA Graph](../../../docs/稀疏MoE与CUDA_Graph.md)。

- 核心算子：单卡 A800，96 组 BF16/FP16、变路由、空命中/集中命中、padding、置换，严格 Dense 参考与 Graph/eager 全部一致。
- 真实 30B TP8/EP8：小 batch 120、大 batch 224、最终回归 64 个 rank-step，完整 logits/KV、padding、置换、strict Dense 对照全部一致。另有 32 个混合 prefill/decode rank-step。
- 分段图：12 个小模型/NCCL 代表拓扑，共 282 个 rank-step，含 DP/TP/EP/PP/DCP 组合、不等长 DP 和空闲 DP。与同拓扑 eager 的 logits/token/KV 全部一致。
- 真实 30B DP2×TP4×EP8：48 个 rank-step，两个 DP 副本分别 3/1、0/2 请求，连续三步，logits/token/KV 全部一致。
- CPU：161 passed、21 skipped、3 subtests passed。CUDA 分段缓存回归：2 passed，包含重放新内容、输出生命周期与 LRU。

这是正确性验证，没有新增吞吐或 HF 质量结论。512 MiB 分段缓存的真实 DP 测试有频繁重新捕获，尚不能宣称该路径提速。完整图与分段图的次数单独记录。

`evidence.tar.gz` 保存逐 rank JSON、命令、源码哈希、所有日志、失败尝试、套件状态、最终 Python 源码快照、部署前备份和部署指纹。无权重或访问凭据。`artifact_manifest.json` 是其 SHA256；`summary.json` 是验证汇总。解包前核对 SHA256，原始失败未被删除。

最终额外修复分段缓存对捕获闭包的持有，防止临时权重 view 的存储过早释放；该生命周期修复补跑了两个 CUDA 缓存测试。各运行原始源码哈希与最终打包源码哈希分别保存，避免混淆。正式服务器目录更新前逐文件比对原发布版指纹，确认没有独立修改；部署后 20 个改动文件逐项校验，详见归档中的 `deploy_receipt.json`。
