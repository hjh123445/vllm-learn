# 当前并行实现与 vLLM 的差异（2026-11-04）

对照官方 commit `d6b48f928f922849db55751ceadbbb55a5a85260`；实际待测官方基线为隔离的 0.13.0。版本不同，必须分别标注。本文描述实现边界，不代表所有功能通过验收。

## 最关键的区别：EP 不是同一个设备坐标

| 方面 | 本项目 | 官方参考源码思路 |
|---|---|---|
| 设备计数 | DP×PP×TP×EP×CP，都是独立维度 | 单 DP world 为 PP×TP×prefill-CP；EP 复用 attention TP/DP ranks |
| DP2×EP4 | 两个独立 EngineCore、两个独立 EP4 专家池 | attention-DP 可与跨 DP 的 EP group 配合，专家池不一定独立 |
| TP+EP | 每专家 intermediate 再按 TP 切分 | 启用 EP 后，MoE 专家沿 EP group 分布；不能默认仍有本项目同样的 intermediate TP partial |
| attention 输入 | 同副本的 EP ranks 当前复制 attention batch | 可采用 attention DP/TP，EP ranks 的来源 batch 不必一致 |
| DCP | 本项目独立 CP 设备轴 | 官方参考 DCP 复用 TP ranks |

官方依据：vllm/config/parallel.py:152,769，vllm/distributed/parallel_state.py:1581,1611,1688。不要把“接口名字类似”写成“拓扑完全相同”。

## EP 数据路径和结果归约

当前 layers/moe.py 的步骤是：

1. router 产生全局 expert ID、weight；扁平化 token/expert route。
2. 按目标 EP rank 稳定排序，交换 send/recv counts。
3. 用可不同的 input/output split sizes 传 activation、expert ID、weight。
4. owner 仅持本地专家，按本地 expert 分段执行 GEMM。
5. reverse all-to-all 返回每条 route 结果，源 rank 逆置换回 `[T,k,H]`，按指定数值策略聚合 top-k。
6. **combine 之后**沿源端 TP group 归约专家 intermediate partial。无需沿 EP 再加一次，否则重复累加。

专家权重是 `[E_local,...]`，EP8 每卡 16/128 专家，而非每卡开完整 128 专家权重。activation 不按最大专家容量 padding。opt-in replicated-input ownership 为恢复复制 hidden 使用等长 all-gather，会有不超过 EP−1 行的边界 padding；这与专家 dispatch 的容量 padding 是两件事。

若每个 source 有 T 个 token、k 个 route、hidden H、dtype B 字节，单个 source 的 dispatch+combine 的 activation 逻辑 payload 约 `2*T*k*H*B`，另有 expert ID、weight、counts；包含本机目的段，不等于实际跨链路字节。默认复制输入会让 EP 个 source 重复路由，group 总量还受该复制因子影响。ownership 减少重复 route，增加恢复 hidden 的 gather，实测目前未获稳定加速，不能仅凭公式宣称优化成功。

四卡 TP2×EP2 合成 oracle 包括空 source、变长、FP32/BF16；它证明被测 primitive 的组合行为。全模型跨拓扑 greedy 仍有分歧；router logits dtype、GEMM M 维变化、BF16 expert 累加和 attention 数学路径正在做同输入对照。重复 greedy 一致不等价于 HF logits parity。

## 为什么尚不等于高效 fused MoE / DeepEP

本项目已有 dispatch/expert/combine 边界，但 _experts_local 仍逐专家执行 PyTorch GEMM，counts.tolist() 要把设备结果交给 Python，dispatch metadata 也分多次 collective。当前 backend 没有 DeepEP 的专用 dispatch/combine 内核、低延迟协议或细粒度计算通信 overlap。函数边界可替换不等于简单换一个 all_to_all 调用就具备 DeepEP。

下一步先以同输入 route oracle 验证数值，再用 profiler 确定 metadata、launch、GEMM、NCCL 等占比。按收益评估 GPU route layout/offset、grouped GEMM、activation/weighted combine 融合；DeepEP 后端需要对应协议、buffer 生命周期、拓扑支持与调度配合。Qwen3-Omni 此文本 MoE 没有共享专家，不能直接套用“共享专家掩盖 all-to-all”实验。

## TP 与 shared memory

CPU 控制消息已经走 shared-memory slot + notification/ACK；GPU TP 目前保留 NCCL。把 GPU tensor 放进 Python multiprocessing.shared_memory 会增加 D2H/H2D。官方还可按设备/消息条件使用 CUDA IPC custom all-reduce 等；这属于 GPU 可访问内存与专用归约协议，不能与 Python CPU shared memory 混为一谈。

## PP、DP 与验收范围

PP 已有层切分、微批、异步 send、通信 stream、event 和 send credit；仍有阻塞 recv 与单消息 worker，Core 窗口最多两条。邻接 P2P communicator 的 opt-in 改动需要匹配负载对照；更改 communicator 不自动消除 fill/drain、串行 host 或 stage 负载失衡。没有训练 backward、没有完整 DualPipe/zero-bubble、没有同卡多 stage 执行。

DP 每副本独立 EngineCore 与 GPU 集合，语义是推理 replica。前端每 step join，仍有慢副本屏障。官方跨 DP 的 EP 要保证 collective 对齐与空 batch 协调，不能把本项目“副本独立”概括为 vLLM 所有 DP/EP 配置。

完整模型精度、官方同机基线、真实多模态 serving、量化 serving 与投机解码均需单独验收。实验记录必须注明源码指纹、模型、版本、输入/输出长度、dtype、计时边界、采样覆盖和失败情况。
