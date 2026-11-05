# EP = DP×TP：拓扑、数据路径与兼容性

2026-11-05。本次取消旧版独立 EP 设备轴，采用与 vLLM 无 prefill-context-parallel 配置一致的 EP 组原则。这里的 DP 是推理请求副本，不是训练梯度 DDP。

## 设备数与进程组

设备数：`DP × PP × TP × CP`。EP 不再额外乘设备。

开启 `enable_expert_parallel=True` 时：

```text
ep_size = dp_size × tp_size
ep_rank = dp_rank × tp_size + tp_rank
global_rank = ((dp_rank × pp_size + pp_rank) × tp_size + tp_rank) × cp_size + cp_rank
```

每个固定 PP/CP 坐标有一个 EP group。关闭 EP 时，MoE 仍使用普通 TP：全专家编号复制，专家 intermediate 按 TP 切。

例如 DP2×PP2×TP2，共8卡：

| global rank | DP | PP | TP | EP rank |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 | 1 |
| 2 | 0 | 1 | 0 | 0 |
| 3 | 0 | 1 | 1 | 1 |
| 4 | 1 | 0 | 0 | 2 |
| 5 | 1 | 0 | 1 | 3 |
| 6 | 1 | 1 | 0 | 2 |
| 7 | 1 | 1 | 1 | 3 |

TP组 `[0,1] [2,3] [4,5] [6,7]`；EP组 `[0,1,4,5] [2,3,6,7]`；PP链路 `0→2、1→3、4→6、5→7`。

对于128专家，EP4每rank持32个**完整专家**：`w13=[32,2I,H]`、`w2=[32,H,I]`。不是32个专家的半个intermediate，也不是在8卡基础上再乘EP4。

官方依据：[vLLM v0.31.0 initialize_model_parallel](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/distributed/parallel_state.py)。包含PCP等配置时官方EP组还可能覆盖PCP；本项目CP仍是独立DCP实验维度，不声称与官方DCP整体相同。

## Attention与MoE分别怎么跑

每个DP有独立EngineCore、scheduler、请求与KV block manager，管理独立GPU集合。Attention在该DP的TP组切Q/K/V头、o_proj归约；不同DP的输入内容和长度都可以不同。

MoE顺序：

1. 对本DP的完整复制Attention输出 `[T_d,H]` 计算router，保留已有完整M形状策略。
2. 在本DP内部按TP rank分配token区间，每token只由一个TP source发出，避免TP倍重复路由。
3. 按目标专家owner排序，跨DP×TP交换变长counts、activation和metadata。所有rank可以有不同source长度，甚至0。
4. owner只持本地专家，对收到的token按本地专家分组计算完整FFN。
5. reverse all-to-allv把每条route结果归还原source，恢复顺序、合并top-k。
6. 只在原DP的TP组gather，恢复该DP的 `[T_d,H]`，供下一层Attention使用。

EP模式没有专家TP部分和，因此**不再在combine后加一次TP all-reduce**。也不能跨整个EP组gather输出：那会混入其他DP请求。

变长dispatch/combine没有最热专家容量padding；TP输出gather为等长接口，仅token ownership边界最多填TP−1行。TP1不需要此gather。

## 为什么需要空批次

DP0有请求、DP1暂时空闲时，DP1仍持有DP0可能选中的专家。DP1不能不运行，也不能只等自己的下一次请求。

离线DPFrontend先检查所有副本状态，只要有一个尚有请求，就同时驱动所有EngineCore一步。没有本地可调度请求的副本提交空schedule；worker遍历本stage实际MoE层，输入 `[0,H]`，仍参加dispatch/expert/combine，接收并执行其他DP发来的route。该副本没有本地Attention/PP激活，不发送虚假的采样结果。

所有副本都必须以相同轮数进入其对应stage的EP collective。空批次也消耗batch ID，PP队列以同一轮次填充与回收，避免一边跳过、另一边阻塞。全部真实请求完成后，前端停止提交，随后所有副本共同drain剩余空批次并shutdown。

## 入口与迁移

单DP，Attention TP8 / expert EP8，共8卡：

```python
core = EngineCore(model, tp_size=8, enable_expert_parallel=True)
```

DP2×TP4，Attention每副本TP4 / expert EP8，共8卡：

```python
frontend = DPFrontend(model, dp_size=2, tp_size=4,
                      enable_expert_parallel=True)
```

DP2×PP2×TP2，共8卡，每PP stage内EP4：

```python
frontend = DPFrontend(model, dp_size=2, pp_size=2, tp_size=2,
                      enable_expert_parallel=True)
```

server目前仍只服务一个EngineCore，支持 `--tensor-parallel-size 8 --enable-expert-parallel`。多DP通过离线DPFrontend验证，**尚未将其接入AsyncEngine/HTTP多副本服务**。

`ep_size`/`expert_parallel_size`旧参数仅保留为派生值一致性检查；不能再通过它创建独立维度。旧 `--ep` 实验脚本改为 `--tp`、`--enable-expert-parallel`；混合矩阵另有 `--dp`。旧ad hoc矩阵的独立EP配置明确拒绝，不静默把旧配置改名后继续比较。

`bench_workload_matrix.py --dp 2 --tp 4 --enable-expert-parallel` 为8卡DP2TP4入口。`capture_parallel_window.py --pp 2 --tp 4 --enable-expert-parallel` 为8卡PP2TP4/每stage EP4入口；该采集入口仍是单DP。

## 文件职责

| 文件 | 改动 |
|---|---|
| engine/process_groups.py | 纯拓扑公式、TP/DP/CP/EP组、各DP内的PP邻接组 |
| layers/parallel.py | 派生EP大小/坐标，拒绝独立EP参数 |
| layers/moe.py | 完整专家权重、TP token ownership、跨DP×TP all-to-allv、TP输出恢复 |
| engine/frontend.py | 各DP EngineCore进程、共同world初始化、全副本驱动、空闲参与、错误传播/超时/任务进程组清理 |
| engine/core.py | 单副本只创建PP×TP×CP workers；空schedule入队；不把空批次当待完成请求 |
| engine/worker.py | local→global rank、派生EP rank、各DP的采样回复、空MoE执行 |
| engine/model_runner.py | 空副本逐层MoE、数值诊断目录按DP区分 |
| engine/pipeline_transport.py | 加入DP rank offset，避免PP跨接到另一个副本 |

## 验证范围和当前限制

CPU/Gloo最终回归59项通过（17.96秒），含三个四进程拓扑DP1TP4、DP2TP2、DP4TP1的完整专家数值oracle。覆盖不同DP输入、空source、全空输入、TP ownership尾部和route总数守恒；还运行实际Qwen tiny模型、scheduler/executor的PP缓存生成测试。

NCCL首轮10项中9通过、1失败：DP2TP2生产BF16生成相对单卡第三token出现分叉。增加匹配DP/TP/PP配置、关闭EP的生产基线后，仍是9通过、1失败。具体输出为EP模式 `[54,60,54,60]`，基线 `[54,60,57,60]`。这说明不能只把差异归因于Attention TP；完整专家与专家TP的归约、计算形状和舍入路径不同，但确切根因尚未证明。没有删除或放宽该严格生成测试。

为避免自由生成分叉后输入不同干扰数值定位，又独立比较16个固定前缀的BF16 logits向量：DP2TP2 EP4对照同配置的专家TP，最大绝对误差0.013671875，16个argmax均一致，满足本测试的 `atol=0.02, rtol=0.03`。它只证明这些合成小模型前缀的容差一致性，**不能推翻上述自由生成失败，也不是30B/HF正确性或任务准确率结论**。原始24个诊断张量、测试源码快照、命令、日志和JUnit都在 `outputs/verified_parallel/ep_coupled_20261105/`，目录README说明每轮含义。

验证还暴露了独立DP world诊断文件相互覆盖的问题：各world的通信rank都从0开始。现已用前端副本编号区分诊断目录与trace文件，同时保留真实通信rank元信息。最终固定前缀验证已确认24个文件被独立保留。最后的trace文件命名与merge优先级调整通过9项本地合并测试；未重新采集GPU trace，不应把旧trace称作本次新拓扑的运行结果。

仍有以下边界：

- 跨DP每轮join是正确性优先的协调，存在慢副本屏障，不是官方完整异步负载协调。
- coupled DP拒绝 `batch_execution_mode=per_request`，否则各DP请求数不同会发起不同次数collective。
- 每个PP stage有独立EP组，只要求相同stage的DP副本按相同层顺序参与；初始化warmup不跨PP发送激活。
- counts需要回CPU产生split sizes；尚无DeepEP、GPU-only动态调度、EPLB或真实MoE通信重叠保证。
- BF16跨batch/TP的数值路径差异仍需单独验收；通过小型oracle不等于30B模型HF全负载对齐。
- 本轮没有30B吞吐实验；旧独立EP的toks/s与trace不能直接标成此新拓扑的数据。
