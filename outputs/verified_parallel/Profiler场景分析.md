# 如何用现有 trace 解释性能

本报告分析真实 Qwen3-Omni Thinker CUDA 轨迹；合成 transport 图不作模型性能证据。打开 [交互阅读器](TRACE交互阅读器.html)，再用对应场景的 `*_gpu_timeline.json` 在 Perfetto 中查看单个 kernel。阅读器中的长条是 kernel 的起止包络，`busy` 是其区间并集；长条内部可以有空白。

## 先验收采样，再解释空白

第一步看八个 rank 的 `kernel_count`，确认实际 CUDA activity 存在。第二步检查每卡负责的全局层：EP8 每卡 L00–47，PP2×EP4 中 GPU0–3 为 L00–23、GPU4–7 为 L24–47。第三步按 microbatch 分别检查，不能把 MB1 的前半层与 MB2 的后半层拼成“完整消息”。第四步确认 `baseTimeNanoseconds` 一致，保留跨 rank 的原始时间，不能各卡独自归零。

旧 trace 存在缺卡/缺层。新 warm 窗口第一条捕获消息在八卡均有完整层归属，但最后一条仍缺尾层。新采样增加尾部 guard，分析器同时输出 `common_complete_messages` 与完整消息范围内的 stage 计算交集。即使层覆盖齐全，也只能证明每层有归属 kernel，不能证明每一个 kernel 均未丢失。

## 场景一：EP8，每层的通信和小算子

选择 `ep8_warm_window`，观察 GPU0 的 MB1：青色 L00 Attention 后面是红色 L00 MoE，依次到 L47。切换 GPU1–7，看同一层的 NCCL 与专家 GEMM 是否对齐。MoE 包含路由、dispatch、专家计算、combine，不能把 MoE range 和其中 NCCL 再相加。

重点追问：GPU 上短 kernel 很密集，但其间有没有 host gap？计数 `.tolist()`、变长 split 准备、排序与 Python expert loop 都可能造成 GPU 等待。仅凭 kernel 名称不能确认某段完全 memory-bound，需要 CPU launch 相关性和 Nsight 指标。

默认实现还在八个 Attention rank 上复制 token batch，同一 MoE route 被八次派发。ownership 模式只保留一个 source owner，最后恢复复制 hidden。预期 route 数从 `8×T×k` 降到 `T×k`，但多了 gather 和小矩阵形状变化。旧端到端数据未证明稳定收益，不能把 route 数降八倍说成速度快八倍。

## 场景二：PP baseline，fill/drain 与 stage 等待

选择 `pp_baseline_window`，同时看 GPU0 与 GPU4。GPU0 上 L23 完成后发 activation，GPU4 才能开始相应 MB 的 L24；与此同时检查 GPU0 是否已在做下一 MB。一次请求的因果链不可能靠新增 stream 消失，需要多个独立在途 microbatch 才有机会填满两级。

调度器按完整 sequence 组成 microbatch，长请求不能任意拆成多个独立请求。单个长 prefill、请求数少于 stage 深度、最后一个 microbatch 排空都可能形成空闲；对尾部采样缺失必须先排除。

整采样窗口两个 stage 的非 NCCL kernel 交集约 **50.04ms**，占较短 stage 计算区间并集约 **20.06%**。这只说明采样期间两级有 kernel 同时执行，不是 SM 利用率，也不是全程 bubble 比率。

## 场景三：异步 send 与 dedicated P2P

`pp_optimized_window` 的跨 stage 计算交集约 **55.34ms / 22.39%**；`pp_dedicated_window` 约 **59.82ms / 25.05%**。这几个数字取自各自短采样，存在窗口边缘和采样开销，不能直接据此宣称优化有效。

同设备计算 kernel 与 NCCL kernel 的区间交集均未观测到。独立通信 stream 只提供执行机会；阻塞 receive、串行 worker、collective 顺序与依赖仍可能限制重叠。实际三轮吞吐矩阵也未证明 PP dedicated/异步方案有稳定收益。可以陈述“有跨设备流水”，不能陈述“同卡通信已被计算掩盖”或“零气泡”。

`pp_dedicated_warm_window` 整窗口约 **19.83ms / 16.82%**，与前几组窗口、消息内容和长度不同，不应拿其百分比作直接优化排名。完整消息内窗的统计见最新 JSON。

## 场景四：长输入与数据并行为什么改变排名

默认数学矩阵的 4K 输入、8 请求中，EP8 为14.07 output tok/s，DP2×EP4为21.68，DP4×EP2为28.17，PP2×EP4为12.61。这个排名与短128请求不同。它与 attention 复制、prefill 工作、每副本 batch 大小和 stage 调度有关，但目前仅凭 throughput 不能把因果全部分配给某一个因素。

`ep8_4k_window` 与 `pp2ep4_4k_window` 存在缺层，后者观察到 stage 计算交集为0，也不能据此断言“长输入永远不流水”。应补全完整窗口，再把长序列 Attention 时间、MoE 路由计数和 stage completion 相关联。

DP frontend 有独立 EngineCore 和 GPU 集，但 `step_all` 等待所有副本。长短混批要按请求长度分组看 TTFT、TPOT、完成时间；平均吞吐变快可能同时出现短请求被慢副本拖住。8请求的 P99 基本接近最大值，不能包装为线上稳定 P99。

## 面试时如何从图走到下一次修改

| 观察 | 可检验解释 | 下一步证据 | 可能修改 |
|---|---|---|---|
| MoE 内 host gap 多 | counts/splits D2H、Python loop | CPU runtime 与 CUDA launch 关联；同步调用次数 | native grouped、设备端路由计数与布局 |
| NCCL 很长 | 对端迟到、排队或有效传输 | 对端同层时间、payload bytes、拓扑/带宽 | 减少重复 routes、减少 metadata、低延迟 dispatch |
| 单卡 kernel 极短 | 小 GEMM、launch-bound | M/N/K、kernel 次数、Nsight Compute | grouped/fused MoE，避免 route 权重复制 |
| stage1 等待 | stage0 慢、receive 阻塞、在途不足 | MB0/1依赖和stage边界 | 均衡切层、预投递 receive、有限队列 |
| 新 stream 没收益 | 依赖/资源/communicator串行 | 实际 kernel交集和端到端消融 | 修调度/协议，或移除无效固定开销 |
| tail 请求慢 | prefill阻塞decode、DP倾斜 | 长度分组、队列时间、逐token时间 | chunked prefill、负载感知路由、异步完成 |

当前 trace 不含完整 SM active、Tensor Core active、HBM/NVLink bytes 指标。因此所有“带宽瓶颈”“占用不足”等归因都应作为待验证假设，不能由白色空隙或 kernel duration 单独下结论。
