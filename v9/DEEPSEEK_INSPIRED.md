# DeepSeek 思路在 v9 推理引擎中的落地边界

## 先区分训练和推理

DeepSeek 训练系统里的 zero-bubble pipeline、ZeRO-1、重计算和 optimizer-state 分片，不能原样搬进 v9：v9 没有反向传播、梯度或 optimizer state。可以迁移的是它们背后的系统思想：让流水线持续有工作、让 EP 通信和独立计算重叠、让专家负载可观测、让 CUDA kernel 减少 Python/ATen 调度。

| DeepSeek 思路 | v9 推理对应物 | 当前状态 |
|---|---|---|
| zero-bubble pipeline | forward-only 多微批 + PP 激活异步发送 | `pipeline_schedule="zero_bubble"` 已加入；这是前向近似，不是训练 ZB-H1/ZB-V |
| EP=8、放弃 TP | `tp_size=1, ep_size=8` | 已支持；专家权重按 EP owner 构造，token 使用变长 all-to-all dispatch |
| ZeRO-1 | 推理没有 optimizer state | 不实现；对应的推理优化是 EP 权重切分、KV 分页和 buffer 复用 |
| overlap shared expert / A2A | replicated shared expert + CUDA side stream | 已加入可选路径；只有配置声明 shared expert 且 EP-only 时启用 |
| fused CUDA kernels | fused RMSNorm、RoPE/cache、SiLU×mul | 已有；专家路径已复用 `silu_and_mul`，router/dispatch 仍需 fused kernel |

## Forward-only zero-bubble 模式

训练 zero-bubble 依赖 forward/backward/backward-input 等不同工作单元的重排；推理只有 forward，因此 v9 采用可验证的近似：

1. 一个 scheduler batch 先分成多个 microbatch；
2. 每个 microbatch 带有唯一的 `microbatch_id` 和 virtual-engine slot；
3. EngineCore 连续发送多个 microbatch；
4. PP 中间 stage 通过 `isend` 发送激活，下一 stage 同时计算前一个 microbatch；
5. `zero_bubble` 模式至少保留 `2 * pp_size` 个 microbatch slot，实际数量受当前请求数限制。

这能减少稳态等待，但不能消除 fill/drain 边界。只有一个请求或请求数少于 pipeline 深度时，仍然会有气泡；因此 benchmark 必须固定请求数和 prompt 长度，分别报告 `m=1`、`m=pp_size` 和 `m=2*pp_size`。

## EP=8 与 shared expert overlap

EP=8 的启动方式是：

```python
EngineCore(
    model_path,
    tp_size=1,
    ep_size=8,
    config_kwargs={"pipeline_microbatches": 8},
    gpus=list(range(8)),
)
```

每个 token-expert pair 根据 owner rank 变长发送。shared expert 权重在 EP ranks 上复制；路由专家执行 all-to-all 时，shared expert 在独立 CUDA stream 上执行。当前只在 `tp_size=1`、CUDA inference 模式启用，避免把未分片的 shared expert 错误复制到 TP ranks。

这条路径必须用 CUDA event 建立依赖：side stream 先等待输入 `x` 就绪，主 stream 执行 dispatch/combine，最后主 stream 等待 shared stream 完成再相加。不能只依靠 Python 语句顺序推断跨 stream 的可见性。

## 仍需实现的高性能部分

当前实现可以验证算法和通信语义，但还不是 DeepEP：

* `all_to_all_single` 前仍有 GPU→CPU 的 `tolist/item` 计数同步；
* hidden、expert id、routing weight、token id 分多次 collective 传输；
* `argsort` 和逐专家 Python loop 没有融合；
* 每次通信动态分配 receive buffer，没有稳定的 workspace；
* 没有 capacity factor、token drop 或负载均衡 loss；
* shared expert overlap 仅覆盖 replicated dense expert，不覆盖 DSV4 专用 runtime。

下一步若要接近 DeepEP，应先实现 GPU-side dispatch layout：一次生成
`num_tokens_per_rank`、`num_tokens_per_expert` 和 prefix offsets，再用预分配 buffer 执行 dispatch/combine。之后再考虑 Triton/CUDA fused router、grouped GEMM 和通信/计算双缓冲。

## 可验证实验

在 A800×8 上固定模型、dtype、prompt 和生成长度，比较：

* EP8：`tp=1, ep=8` 与单卡/TP8 的输出一致性、每层 dispatch 时间、all-to-all 时间；
* shared overlap：关闭和开启 side stream，比较 MoE layer latency 与 CUDA event 时间线；
* PP：`pp=4` 时 `m=1/4/8` 的 bubble、TTFT、TPOT、吞吐；
* imbalance：人为构造路由集中到一个专家，记录各 rank 接收 pair 数、峰值显存和 tail latency；
* kernel：PyTorch SiLU×mul 与 `silu_and_mul` 的数值误差和 kernel time。

没有 Nsight Systems/CUDA 实测时，只能说明路径已经接入，不能声称通信被完全隐藏或达到了 zero bubble。
