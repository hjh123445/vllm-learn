# PP×EP、Virtual Engine 与一张卡多 Stage

## PP×EP 是否是更好的通用选择

对 MoE 推理，PP×EP 通常比单独 PP 或单独 TP 更自然：

- PP 沿层切分，每张卡只保留一段 decoder layer 和对应 KV cache；
- EP 沿专家切分，每个 stage 内的专家只加载到所属 EP rank；
- token 在每个 stage 的 MoE 层通过 EP dispatch/combine 到专家 owner；
- attention 的 TP/EP/PP 通信域彼此独立，避免把跨层 activation 与专家 token
  dispatch 混成一个通信组。

但它不是无条件最优。PP 引入 fill/drain bubble，EP 引入 all-to-all 和负载长尾；
最佳配置取决于 prompt 长度、decode batch、专家负载方差、GPU 拓扑和显存。Qwen3
Omni-30B 这类 MoE 推理通常先用 EP 降低专家权重复制，再用 PP 解决单卡显存，
最后才考虑 TP；必须用吞吐、TTFT、TPOT、通信占比和热点专家长尾实测决定。

## Virtual Engine 的真实含义

本版 VirtualEnginePool 是显式的 microbatch lane 所有权：

- 每个 slot 有稳定的 lane；
- microbatch 提交时增加 in_flight；
- 最后 stage 返回后按 slot 完成；
- slot 生命周期独立于 EngineCore scheduler 的 sequence/KV 所有权。

它已经接入 EngineCore，不再只是消息字段。pipeline_lanes=2 时，microbatch
0/2 复用 lane 0，1/3 复用 lane 1；credit 窗口限制同一 lane 的 activation
滞留数量。

这和生产 vLLM 的 virtual engine 仍有边界：vLLM 还会在 virtual engine 上维护
请求队列、KV block 视图、异步输出和抢占状态。本项目的 slot 是推理 PP 调度的
可验证子集，代码中明确区分了两者。

## 一张卡承担多个 Stage 能否实现

可以，叫 interleaved PP 或 virtual pipeline stage。例如 4 个逻辑 stage、2
张物理卡可以规划成：

logical stage:  0  1  2  3
physical card:  0  1  0  1

规划器已经在 engine/pipeline_schedule.py 中实现了
PipelineTopology.interleaved(pp_size, workers)，会验证每个 worker 都有 stage
并给出 stages_for_worker()。

当前 Qwen3-Omni worker 仍是一进程一个 STATE.pp_rank 和一个模型 stage，因此
运行时对“同卡多 stage”会显式抛出 NotImplementedError，不会悄悄把它当成普通
PP。要真正启用，需要下一步实现 local-stage executor：

1. 一个 worker 加载同卡多个不重叠 layer range；
2. 同卡相邻 stage 直接传递 GPU tensor，不走 NCCL；
3. 只有物理卡之间的边界使用 P2P；
4. 每个 local stage 有独立 KV cache slice、trace stage id 和 microbatch
   mailbox；
5. stage 0/2 的计算需要同一张卡上的 CUDA stream 或串行依赖，避免两个 stage
   同时争用 SM 和显存。

因此“一张卡多 stage”可以实现，但它不是把 pp_rank 改成数组就结束；显存
收益有限，主要用途是用更多 virtual stage 换取更细的层间切分和更好的负载均衡。
当每个 stage 已经接近满载时，同卡多 stage 反而可能增加上下文切换和调度开销。

## baseline / optimized 时间轨迹

EngineCore 支持 pipeline_mode=baseline 或 optimized：

- baseline：header、hidden、residual 使用阻塞 P2P；
- optimized：异步 isend/irecv、通信 stream、CUDA event、按 lane 懒投递
  header receive、per-lane send credit。worker 仍逐条处理消息，recv 阶段会等待。

设置 VLLM_LEARN_TRACE_DIR=/path/to/dir 后，每个 worker 会生成
pipeline_rank<N>.json。Chrome trace 中：

- stage<N>.layer<M>.attention / .moe：带全局层编号的模型算子区间；
- stage<N>.microbatch<M>.lane<L>.prefill/decode：一次 worker 消息；
- pp.send / pp.recv：P2P activation；
- args.layer_range：该 stage 的全局层范围；
- args.microbatch_id、args.lane：微批和 virtual lane；
- metadata 中的 pipeline_mode：baseline 或 optimized。

两种模式必须使用相同输入、微批 token budget 和 GPU。Chrome trace 的 GPU
kernel stream 可以判断采样窗口内是否出现 compute/NCCL 区间重叠；少量活跃
step 的 profile 不能估算完整请求的 pipeline bubble ratio。



## 本轮实现校正

- EP 的变长 dispatch/combine 会把各目的卡的专家输出返回原 token 所在卡；因此 all-to-allv 路径只需要 TP 归约，不能再对 EP 维度重复 all-reduce。
- VirtualEnginePool 现在真正参与 EngineCore 的微批提交和完成记账，并把 lane、microbatch_id、virtual_engine 回传到 worker；它是 forward-only lane ownership，不能冒充完整 vLLM scheduler virtual engine。
- `build_forward_schedule` 仅提供 CPU 测试用的理想时钟计划，不是 runtime scheduler 或真实执行轨迹。
- layer trace 现在会标出 `attention`、`moe`、全局 layer id、PP stage、microbatch 和 lane。设置 `VLLM_LEARN_TRACE_DIR` 会输出这些主机时间线；再设置 `VLLM_LEARN_CUDA_PROFILE=1` 会额外输出 `cuda_rank<N>.json`，其中包含 CUDA/NCCL 活动（profile 采用 1 个等待、1 个 warmup、8 个 active step，避免无限增长）。

## 同卡多 stage 的边界

`PipelineTopology.interleaved(4, 2)` 可以准确表达 `(stage0, stage1, stage2, stage3) -> (card0, card1, card0, card1)`，并可用于规划和测试。当前执行器仍是一进程一模型 stage，因此 EngineCore 会在收到这种配置时立即抛出 `NotImplementedError`，避免把逻辑 stage 映射错误地当成普通 PP 运行。真正接通需要 worker 在同一进程中持有多个 stage model chunk，增加本地 activation mailbox，并让只有跨物理卡的边界使用 NCCL P2P；这一步不能只改一个 stage map 字段。

`VirtualEnginePool` 当前只管理 forward microbatch 的 lane/in-flight 记账，没有
vLLM virtual engine 的完整请求队列、KV block 生命周期、抢占与异步输出语义。
旧配置中的 `pipeline_microbatches`、`virtual_engine_size` 和 `--zero-bubble`
没有进入 runtime 调度，现已移除，避免选项看起来生效但实际无效。实际分批由
scheduler 输出及 `pipeline_microbatch_tokens` 决定。
