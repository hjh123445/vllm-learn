# P0 修复说明（fix/p0-interview-hardening）

> 基线：`36f5e4b`。本分支修复全面代码审查定位的 18 个 P0 级问题
> （"面试现场一票否决"级：跑不起来 / 死锁 / 静默错误输出）。
> 每项修复都带 `P0-x` 注释锚点，与审查报告编号一致。
> **验证状态**：25 个改动文件全部通过语法编译；`_Spec` 切片协议通过纯逻辑
> 内联验证；GPU 级验证（T1-T7 回归 + accuracy + matrix 冒烟）需要在
> A800 机器上跑（见文末清单）。

## 修复清单

### v9 引擎（5 项致命）

| # | 文件 | 问题 → 修复 |
|---|---|---|
| P0-1 | `engine/model_runner.py` | **任何模型都无法启动**（双重）：(a) `import os` 缺失——Qwen 路径的 `os.path.basename` 直接 NameError；(b) `load_thinker_config()` 无条件执行在模型分派之前——非 Omni 模型没有 `thinker_config` 键，KeyError。→ 加 import；新增 `_load_hf_config()` 按模型类型分派（Omni 平铺 text_config，其余用 AutoConfig + config.json 平铺），`_is_dsv4()` 收拢判定 |
| P0-2 | `engine/core.py` | **chunked prefill/混合批双重切片**：`_spec_msg` 已按 `num_cached_tokens` 切好 chunk 装进消息，worker 侧 `prepare_prefill` 又对短列表用全局下标 `[start:end]` 再切一次——第二轮 chunk 及混合批的 decode 行全部切出**空输入**（所有现有实验恰好只走首 chunk 全 prefill 批 + 纯 decode 批，从未触发）。→ 切片责任唯一化：消息携带 `chunk_start`，`_Spec.__getitem__` 做全局→本地翻译，越界抛 IndexError（协议破坏第一现场可见） |
| P0-3 | `engine/worker.py` + `core.py` | **response 背压死锁**：非 PP 时每个 worker 每步都 send，core 只 recv rank0 的——rank1+ 的消息在 Pipe ~64KB 内核缓冲区无限堆积，几百步后 worker 阻塞在 send → 整机死锁（实验只跑 48 步未暴露）。→ 对齐 vLLM `output_rank` 模式：`resp_rank` 放进广播消息，只有命中 rank 回包；ready 路径保持原语义（非 PP 全 rank 发、core 取 min） |
| P0-16 | `engine/scheduler.py` | **自抢占后留在本轮 scheduled**：池满时抢占自己后继续按陈旧的 `num_sched` 调度，decode 读从未写入的 cache → 垃圾输出。→ 抢占自己（两种形态）立即 `return False`，下一轮从 waiting 重新准入重算 chunk |
| P0-18 | `engine/core.py` | **`generate()` 永远返回空列表**（stub 丢掉全部输出）。→ 收集 step() 完成清单 + Sequence 兜底，返回 `[{token_ids, finish_reason}]` |

配套（`engine/scheduler.py`）：`add()` 加准入校验——超 `max_model_len` 或超池容量的请求在**入队期**拒绝（ValueError），不再落到 `assert scheduled_seqs` 崩溃 / v2 式空转死循环。对齐 vLLM `_check_enough_kv_cache_memory`。

### v9 量化（2 项）

| # | 文件 | 问题 → 修复 |
|---|---|---|
| P0-10 | `layers/quantized.py` + `models/qwen3_moe.py` | **量化+TP 静默错误**：`quantize_int4` 把 o_proj（RowParallelLinear，forward 内置 allreduce）换成无通信的 QuantizedLinear——TP>1 时每卡只有 partial sum 永不归约。→ 新增 `QuantizedRowParallelLinear`（保持 K 列切分 + weight_loader 吃全量 + forward 带 `tp_allreduce`），o_proj 用它替换 |
| P0-14 | `layers/moe.py` | **EP+量化直接 AttributeError**：EP 路径的 `_experts_local` 访问已被 `del` 的 `w13/w2`——旗舰实验配置（EP×W4）不可运行。→ quantized 分支按**命中的专家**逐个反量化（B1 数学正确性优先，峰值也低于全量反量化）；v9.2 kernel 化后换 per-expert dequant_gemm |

配套（`layers/quantized.py`）：(a) buffer 宽度按 `padded_k` 分配（非 group 倍数的 in_dim 不再 copy_ mismatch——潜伏 bug）；(b) `rtn_quantize` 的 scale clamp 从 1e-10 提到 6e-5（fp16 scale 对极小组的下溢保护）。

### v0/v2/v4（3 项静默错误）

| # | 文件 | 问题 → 修复 |
|---|---|---|
| P0-13 | `v0/engine/batch.py` | **幻影 token**：`collect()` 对 "length" 行不截断——静态批跑到最慢请求结束，先完成行返回整批 gen_len 的切片（含死 token）。**数据证据**：`speed_v0.json` 记录 1920 tok vs `speed_v1.json` 的 1152（真实值）。→ 每行完成时刻记 `done_len`，collect 按它截断。**⚠ README 性能矩阵的 v0 吞吐列（155.6）口径失真，需重跑** |
| P0-4 | `v2/engine/scheduler.py` + `kv_cache.py` | **off-by-one 写坏 KV**：`ensure_append` 传 `cur_len` 而非写位置 `cur_len-1`——`len(prompt)%16==0` 时跨块不领新块，`block_table` 补零把缺块指向 block 0 → 静默覆盖 prompt 首 token 的 KV（accuracy 测试的自然语言 prompt 恰好都不是 16 倍数，全绿掩盖）。→ 传写位置；**补零逻辑改 assert**（内存 bug 第一现场可见） |
| P0-5 | `v4/engine/model_runner.py` | **池预算陈旧 peak**：ENGINEERING_NOTES §2.4 的"先释放再量"修复在 v4 上算术不生效（`peak-current` 含已释放的 T，预算同样小 T/block_bytes）。v9 已是正确写法，回移植：释放 tmp 后 `reset_peak_memory_stats()`，allocate 直接用 `_activation_peak` |

### v5 在线服务（1 组）

| # | 文件 | 问题 → 修复 |
|---|---|---|
| P0-6 | `engine/async_engine.py` | 三处：(a) `_loop` 零异常处理——任何异常让 daemon 线程静默死亡、全部请求永久挂死 → 整体 try/except + `st.error` 广播 + `_dead` 快速失败（对齐 vLLM ENGINE_CORE_DEAD 语义）；(b) `add()` 发布竞态（先入队后注册 state，一步完成的请求永久挂死）→ `Sequence.peek_next_id()` 预读号 + state 先注册；(c) states 只增不删 → finish 后 `del`；空批分支加 sleep 防热自旋 |
| P0-7 | `server.py` | SSE 从未发送 `data: [DONE]`（docstring 声称发；serve_bench 的 `[DONE]` 分支是死代码）→ 补发；错误传播（流内 error 事件 / 非流式 500）；`/health` 改报线程存活（旧版报"队列非空"，引擎死亡时反而报健康）；`do_POST` 参数解析防护（坏输入回 400 而非裸断连） |

### v6 并行（2 项）

| # | 文件 | 问题 → 修复 |
|---|---|---|
| P0-8 | `tests/parallel_worker.py` + `layers/moe.py` | **TP×EP 组轴向反了**：`tp_group` 按 `r//ep` 分组——那恰好是 EP 组的成员集，attention 部分和凑不齐 tp 个头切片，输出必错（文档归因"allreduce 重复相加"只对一半）。→ tp 组改 `r%ep`；新增 `ep_group` 注入 MoEMLP（EP 聚合限定在同 tp_rank 的 ep 成员上，消除与 tp_allreduce 的重复累加——open issue #3） |
| P0-9 | `tests/pp_worker.py` | **PP 两个真 bug**：(a) p2p 收发不配对（发 2 条收 1 条，residual 被扔）→ NCCL 按提交顺序配对，第 2 步起消息静默错位——这很可能才是"lockstep 死锁"的真因（文档归因 scheduler 分叉不对）→ 接收端补第二条 recv；(b) 末 stage `norm(x)` 丢残差（注释声称"已含残差"是错的）→ 换 `fused_add_rmsnorm(x, residual, ...)` |

### v8 / DSV4 / 测试（3 项）

| # | 文件 | 问题 → 修复 |
|---|---|---|
| P0-11 | `v8/layers/mtp.py` | v8 目录躺着带 bug 的 v7.0 旧版（`if t != 0 or k > 1` 丢 token_id==0 的合法 token 等），版本叠加的目录同步失效 → 拷入 v7.2 正确版（含 mtp_impl.py） |
| P0-15 | `v9/models/deepseek_v4.py` | compressor 从未接入 forward（`n_comp = L_full - T` 恒 0）——完整实现需要 prefill/decode 双 cache 语义，超出本分支范围 → **诚实降级**：类 docstring + forward 内注明"未完成项、当前是单发模式、1.7 tok/s 数据来自 v6.3 引擎"，防止被当已实现宣讲 |
| P0-17 | `v9/tests/test_mixed_batch.py` | "chunk 无关性裁决"是**平凡通过**（prompt ~100 token << 预算 1024，两 phase 路径相同）——这正是 P0-2 的防线失守原因 → prompt 拉到 3000+ token，逐步记录 chunk 轮数与混合批轮数，**断言 chunk 真的发生了**（否则报"测试无效"） |

### 新增：`v9/tests/test_p0_regressions.py`

七个纯逻辑回归测试（子进程隔离，与 benchmark 同机制）：T1 v0 幻影 token 截断、T2 v2 off-by-one（含补零改 assert）、T3 `_Spec` 切片协议（含越界 raise）、T4 准入校验、T5 `peek_next_id` 语义、T6 量化 padded 宽度、T7 RTN scale 下溢。本机（无 GPU/torch）已验证：全部文件 `py_compile` 通过 + `_Spec` 逻辑内联验证（旧版双重切片产出 0 长空输入、新版正确翻译）。

## ⚠ 需要在 GPU 机器上完成的验证（合并前）

1. `python v9/tests/test_p0_regressions.py` —— 7/7 PASS；
2. `python v9/tests/test_mixed_batch.py` —— 断言 chunk 发生 + 两 phase 输出一致（这是 P0-2 的直接裁决）；
3. `python benchmark/accuracy.py` —— T1/T2/T4 仍全绿（v0 的 T2/T3 输出会因 P0-13 修复而**变化**：length 行不再多出幻影 token，v0 vs v1 等价性反而应该提升）；
4. **重跑 v0 的 speed benchmark**（P0-13 修复改变 v0 的 token 口径：1920 → ~1152）并更新 README 性能矩阵的 v0 行；
5. TP2 短冒烟（验证 P0-10 的 allreduce 语义）：同一 prompt TP1 vs TP2 的 logits 对比；
6. PP2 短冒烟（验证 P0-9）：之前从未通过的"PP 输出 vs 单卡一致"这次应该能过——如果仍不过，按 residual 收发路径继续查。

## 明确不做（记录边界）

- P0-12（dsv41_a100_runtime 的 vendoring 补齐）：0.5-2 天的独立工作，涉及 13 个缺失模块的上游同步，不混入本分支；
- DSV4 compressor 的完整实现（P0-15 只做声明降级）：需要 prefill/decode 双 cache 语义设计，属于 v9.x 的功能开发；
- CUDA graph / 联合决策矩阵实验：那是审查报告 §6.2 的深挖主线（前置修复正是本分支）。
