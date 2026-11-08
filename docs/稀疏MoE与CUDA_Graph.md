# 稀疏 MoE 如何进入 CUDA Graph

2026-11-08。本次增加 GPU 路由的稀疏 EP decode，以及跨现有并行方式的计算分段 Graph。以前的 Dense-EP 测试结果仍归属于原源码和原执行策略，不能直接记到新路径上。

## 1. 先区分三条执行路径

| 路径 | 专家计算 | 通信 | 当前 Graph 支持 |
|---|---|---|---|
| 普通 EP，含 DP>1 | 仅路由命中的专家 | counts 交换、all-to-allv dispatch、reverse all-to-allv、本地 top-k 求和、TP all-gather 恢复布局 | 计算分段 Graph，动态通信在图外 |
| 新增 DP1 稀疏 decode | GPU 压紧本地命中 token，仅计算非空专家 tile | 普通数值策略 FP32 all-reduce；严格策略 all-gather 后固定 rank 顺序 FP32 求和 | 支持完整语言 decode Graph |
| Dense-EP 消融 | 本地每个专家计算每个 token，再乘路由权重 | 同上 | 支持，保留显式消融入口 |

普通未开启 Graph 的引擎仍走原 all-to-allv。`benchmark.py --decode-graph --graph-mode auto` 在 DP1/PP1/CP1/EP>1 时选择完整稀疏 decode 图；其他拓扑选择计算分段图。直接设置 `VLLM_LEARN_DECODE_GRAPH=1` 也按同样规则选择。Sparse/Dense 两种专用全图计算开关不能同时为 1。

DP1 专用全图稀疏 decode 要求 EP>1、PP1、CP1，CUDA BF16/FP16 非量化专家、Triton attention。DP1 的 attention 输出已经在 TP ranks 间复制，无需再 all-gather 输入；DP>1 各副本输入不同，使用下述通用分段图保持原 dispatch/combine。图像仍在 prefill 阶段编码，图中捕获语言 decode；既有多模态 PP/DCP 限制不因增加 Graph 而自动消失。

## 1.1 所有并行方式如何接入：完整图与分段图区分

`VLLM_LEARN_DECODE_GRAPH_MODE=auto|full|piecewise`：

- `auto`：符合上述静态条件时使用 full，否则走 piecewise。
- `full`：明确要求完整语言前向图，遇到尚不支持的动态通信会报错。
- `piecewise`：可用于单卡、TP、DP、EP、PP、DCP 及其合法组合；每个 GPU 独立捕获纯计算段，原有通信、调度和主机动态处理在图外执行。

分段实现为 `v9/utils/piecewise_graph.py`。捕获的段包括线性投影、router、每个非空专家的两个 GEMM+SwiGLU、非 DCP 的 decode attention，以及 DCP 的本地 partial attention。TP 归约、EP 变长 all-to-allv、DCP partial 合并通信、PP P2P 继续使用原协议。

每一段以输入形状/dtype/device、权重地址和计算策略作为缓存 key。第一次遇到该形状时预热并捕获；以后把新内容复制到静态输入并 replay。专家收到 3 条路由和 12 条路由对应不同形状的两个缓存条目，而不是强行沿用旧 splits。缓存用 `decode_graph_reserve_bytes` 预算、LRU 淘汰；过大的单段明确报错。新形状的 capture 代价、静态输入/输出复制、动态通信和主机同步仍然存在，不能据此宣称一定提速。

DP 的各副本不需要同一计算 shape，因为这些计算图中没有 collective；EP collective 仍必须按各层相同次数参与，空闲副本仍执行远端路由。PP transport 在图外按 microbatch id 传 hidden/residual，stage 内的计算段独立重放。DCP 在图外构造本地 KV 索引和可见性 mask，按实际本地 KV 长度选择 partial attention 段，然后执行原 all-gather/LSE 合并。

执行记录分别保存 `graph_mode`、`graph_segment_captures/replays`、`captured_buckets`、`graph_capture_count/replays`，防止把分段重放次数当成整模重放次数。当前分段路径不消除动态通信开销，也不是 DP/PP/CP 整个分布式前向的一张大图。

## 2. 默认 MoE 为什么不能原样捕获

例如 token 数固定为 8、top-k=8，总路由条目固定为 64；但本 rank 接到的路由数可能这一步是 3、下一步是 12。默认路径会：

1. 在 GPU 生成目标 rank 的 `send_counts`。
2. 用一次 `all_to_all_single` 交换 counts。
3. 调用 `send_counts.tolist()`、`recv_counts.tolist()`，将 counts 读回 CPU。
4. CPU 用 `sum(recv_counts)` 分配接收张量，把 Python split 列表传给变长 all-to-all。
5. 默认 loop 专家后端还读取各专家 offsets，决定 Python 循环的切片边界。

这些是路由内容决定的主机控制流和分配。Graph replay 不重新运行 Python，也不会重新读这些 split 列表。不能捕获一次后继续沿用旧 counts，否则新的路由会导致错误的通信和索引。仅把专家 GEMM 换成 grouped_mm 也没有解决前面的动态通信问题。

CUDA Graph 要固定的是捕获的 kernel/通信操作及其参数、缓冲区地址和依赖关系；**它不要求缓冲区中的 token、路由索引、权重和上下文长度不变。** GPU kernel 内部可以读取新的 counts 并据此执行分支或循环。这是本次实现的关键。

## 3. 外部 batch 怎么固定

初始化时捕获 1/2/4/8/16/32/64/128/256/512 中不超过 `max_num_seqs` 的 bucket，并加入 `max_num_seqs` 本身。实际 5 个 decode 请求选 bucket 8。

每个 bucket 持有固定地址的 `ids[B]`、`pos[B]`、`slots[B]`、`lens[B]`、`block_tables[B,max_blocks]`。三行 padding 使用 dummy token、长度 1、`slot=-1` 和专用 dummy KV 块，避免覆盖真实 KV；返回只取前 5 行 logits。padding 行仍可能产生路由和计算，但不能影响有效行结果。

每步先把新数据复制到这些相同地址，再 `graph.replay()`。图中重新执行 router、专家选择、attention、专家计算、通信和 logits；不会冻结初始化时的专家选择。采样、请求调度和外部输入填充不在这个捕获范围内。

## 4. 内部专家 token 数怎么可变

实现位于 `v9/layers/sparse_decode.py`。例子使用 Qwen3-Omni Thinker，B=8、H=2048、I=768、E=128、k=8、EP8，单 rank 有 L=16 个完整专家：

| 张量 | 形状 | 内容 |
|---|---|---|
| x | [8,2048] | TP ranks 已有相同的输入 |
| topk_idx / topk_w | [8,8] | 每 token 的 8 个专家及 BF16 路由权重 |
| w13 / w2 | [16,1536,2048] / [16,2048,768] | 本地完整专家权重 |
| tokens | [16,8] int32 | 每专家命中的 token 编号，只有前 counts[e] 项有效 |
| counts | [16] int32 | GPU 上的本地专家命中数量，每次重放重新生成 |
| route_w | [16,8] FP32 | 按原 token 行号索引的路由权重，未命中为 0 |
| h13 | [16,8,1536] | 压紧路由的 gate/up 输出，仅有效前缀可读 |
| act | [16,8,768] | 压紧路由的 SwiGLU，仅有效前缀可读 |
| weighted | [16,8,2048] | 散回原 token 位置的 BF16 加权贡献，未命中为 0 |
| local / final | [8,2048] | 本地 FP32 求和，再跨 rank 合并、最后转回 BF16 |

处理分四步：

1. 每个本地专家扫描 GPU 上的 top-k 表，生成 hit mask；`cumsum` 得到压紧位置，写 tokens 和 counts。top-k 专家唯一，同一专家每 token 最多一个有效条目。
2. 第一 GEMM 用压紧的 token 编号从 x gather 输入，输出 gate/up。启动网格固定；每个 tile 在 GPU 读取 counts，`start>=counts[e]` 时跳过矩阵计算和权重读取。
3. 对有效前缀执行 SwiGLU，再执行第二 GEMM；第二 GEMM 输出先舍入为 BF16，再乘 BF16 路由权重，按 token 编号写回 weighted。
4. 按本地专家编号递增用 FP32 累加；严格策略再收集各 rank 的贡献，按 rank 顺序用 FP32 累加，最后转回模型 dtype。

**预留 [L,B,...] 容量不等于执行 L×B 个有效专家计算。** 未命中的专家不进入 GEMM 分支。矩阵 tile 是 BM=16，因此最后一个非空 tile 内可能有 masked lanes；这属于常规 GEMM 对齐开销，不能声称 FLOPs 恰好等于有效路由数。当前也仍有固定网格空 tile 的检查开销和较大工作区，尚未实现生产级的持久化专家工作队列。

容量上界是每专家 B 个 token；top-k 唯一保证不会超过这个上界，不设小于真实路由数的容量，不丢弃热点专家的 token。未初始化的中间张量尾部始终被 mask，不参与有效输出。weighted 每次清零，避免上一步命中结果残留。

## 5. 为什么仍保留严格数值策略

从 Dense 改为稀疏后，专家输入行顺序/数量会变化；普通 BF16 GEMM 可能选不同算法。新内核固定 BM=16、BN=64、BK=32，固定 K 累加顺序和 BF16 舍入边界，与原 strict Dense 原语对齐。路由加权、本地专家求和、跨 rank 求和也保持同一顺序。

严格策略跨 rank 实际执行的是 **all-gather + GPU 固定顺序求和**，不是 NCCL all-reduce。它的通信/工作区代价继续存在。本次消除未命中专家的矩阵计算，并未消除严格求和的代价。普通策略可以 all-reduce，但不同策略不能混作同一组精度/性能数据。

## 6. 多卡图怎么捕获

每个 GPU worker 在自己的 CUDA context 捕获本地图；不是一台机器生成一份跨卡 Graph。以 EP8 的 bucket 8 为例，8 个 rank 各捕获一个图，collective 节点按一致顺序匹配。每个 bucket 捕获前后做 barrier；重放时所有相关 rank 都必须选择同一 bucket、执行同一组 collective。

例如 max_num_seqs=256 时 9 个 bucket，每卡 9 个 Graph，共 72 个本地图对象。一次 decode 选择其中一个 bucket 对应的 8 个图共同重放。跨机器原则也是各 GPU 单独捕获、通过通信同步，但本项目此次没有验证多机 Graph。

## 7. 如何复测

```bash
python v9/tools/validate_sparse_decode.py --output core.json
python v9/tools/validate_decode_graph.py --model "$MODEL" --output sparse_graph \
  --max-batch 8 --max-model-len 512 --prefill-tokens 1024 \
  --batches 1 3 4 7 8 --lengths 17 255 256 --steps 3 \
  --permute --mixed-lengths --random-inputs \
  --batch-invariant --strict-batch-invariant --compare-dense
```

核心检查在单张 A800 执行 96 组：BF16/FP16、H=80/I=48 的非整 tile 形状及真实 H=2048/I=768、B=1/3/8/17/32/65、四组变化路由，包含全部非本地、集中到同一本地专家。所有路由贡献逐位等于 strict Dense 同权重参考，Graph 重放、置换、padding 也逐位相同。

真实 Qwen3-Omni Thinker 全模型 TP8/EP8/DP1：B=1/3/4/7/8、混合前缀 17/255/256、连续 3 步，8 ranks 共 120 个 rank-step；完整 logits、KV、跨 rank、一致输入的 unpadded eager/padded eager/Graph、请求重排、strict Dense 对照全部逐位一致。每 rank 有 30 次 Graph replay；capture 哨兵验证有效 KV 未被污染。

原始结果位于 `outputs/validation/sparse_graph_20261108/`。这些检查不是新增 HF 质量评测，不应把历史 ChartQA 的 250/256 HF 一致率直接归给新路径，也不能用旧 Dense 吞吐当新稀疏吞吐。

DP1 大 batch 补充：真实模型 B=1/8/9/16/17/32/33/64/65/127/128/129/255/256，混合前缀 17/255/256/511，连续两步、置换、strict Dense 对照，共 224 个 rank-step，全量 logits/KV 逐位一致。

分段图小模型/NCCL 矩阵使用独立合成 checkpoint（H=64、head_dim=16、8 层、4 专家、top-2）。12 个合法组合：单卡、TP2、TP2+EP2、DP2+EP2、DP2×TP2+EP4、PP2×TP2+EP2、DP2×PP2+EP2、TP2×DCP2、TP2×DCP2+EP2、DP2×TP2×DCP2+EP4、DP2×TP2×PP2+EP4、PP2×DCP2。每组覆盖 DP 不等长和仅最后一个 DP 副本有请求、每轮三步 decode，全部通过同拓扑 eager 的完整 logits、token 和 KV 逐位对照。首轮 fixture head_dim=4 不满足 Triton prefill dot 的 K>=16 约束，已改为有效 head_dim=16；另对空闲 KV 槽位先清零再作整池比较，避免未初始化 NaN 造成伪失败。两次失败日志保留。

这些是代表性拓扑回归，不是对任意并行度、量化方式、多机和所有负载的穷举保证。

真实 30B Thinker 的 DP2×TP4×EP8 分段图：第一轮两个 DP 副本分别 3/1 请求，第二轮分别 0/2 请求，前缀包含 17/255/256，每轮连续三步；8 ranks 共 48 个 rank-step。完整 logits、greedy token 和整池 KV 与同拓扑、同严格数值策略的 eager 逐位一致，且记录了实际 segment capture/replay。分段缓存预算设为每卡 512 MiB；这不是吞吐测量。

最终 CPU 套件 161 passed、21 skipped、3 subtests passed；两个真实 CUDA 分段缓存测试通过，覆盖新内容重放、返回结果不被下次 replay 覆盖、形状缓存 LRU 淘汰。skip 不等于 GPU 通过。最终全图入口还补跑了 B=1/3/7/8、两步 decode 的 64 个 rank-step，保持 logits/KV/strict Dense 逐位对照通过。上述原始证据含每次命令、逐 rank JSON、日志和源码哈希。
