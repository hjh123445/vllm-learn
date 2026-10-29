# v7：投机解码（Speculative Decoding）

建立在 **v6**（多卡并行引擎）之上——版本叠加，目录 = v6 全部 + 投机解码组件。

## 小版本演进

| 小版本 | 主题 | 状态 | 独立文档 |
|---|---|---|---|
| v7.0 | n-gram draft + verify 循环（正确性验证） | ✓ | 本文 §实现记录 |
| v7.1 | MTP 层 draft（Qwen3 自带权重，零训练） | 待做 | — |
| v7.2 | 增量 verify（KV 轮间持久，性能反转） | ✓ | 本文 §v7.2 |
| v7.3 | 动态 draft 长度（接受率滑动窗口自适应 k） | 待做 | — |

## v7.0 实现记录（n-gram draft + verify）

**架构**（`layers/mtp.py`）：
```
1. draft:   NGramDrafter 在上下文找 n-gram 前缀匹配 → k=4 个候选
2. verify:  候选拼进序列，一次全量 prefill 取 [L-k-1:-1] 位置的 logits
            （logits[j] 预测位置 j+1——候选 i 的 target 分布在 L-k+i-1）
3. 接受:    从左到右，target argmax == 候选则接受；分歧处用 target token
4. k=0:     draft 无匹配时退化为普通一步（防零产出死循环）
```

**验证**（Qwen3-Omni-30B，48 token greedy）：

| 请求 | 输出与基线一致 | 接受率 |
|---|---|---|
| 数数（1 2 3...） | ✓ 逐 token | 1%（模型转述而非复读——n-gram 匹配不到） |
| 重复这句话三遍 | ✓ 逐 token | **31%**（复读场景 n-gram 命中） |
| 普通文本 | ✓ 逐 token | 0%（无重复结构，预期） |

**当前是慢的（30s vs 基线 4s）**——verify 用全量 prefill（每轮重算整个序列），
这是简化实现。**性能反转点在 v7.2**：接 APC 后 verify 只算 k+1 个新位置，
每轮成本从 O(L) 降到 O(k)。加速比公式：`(α+1) / (1 + k·c_draft/c_target)`。

**踩坑**（四个，全记录）：
1. **零产出死循环**：draft 空 + 全接受时 n_out 不涨 → while 卡死 25 分钟；
2. **verify logits off-by-one**（投机解码第一坑）：logits[j] 预测 j+1 位置的
   token，验证候选 cand[i]（在位置 L-k+i）要用 logits[L-k+i-1]——
   取 `logits[-k:]` 是预测了候选之后的位置，输出整体错位；
3. **脏状态-序列**：手动驱动的 seq 滞留 waiting 队列（cleanup 只查了 running）
   → 下一请求 CUDA assert；
4. **脏状态-cache**：verify 手动写 cache 后 cur_len 未复位 → 下一请求
   attention 越界。

## v7.2 实现记录：增量 verify（性能反转点）

**v7.0 的 verify 为什么慢**：每轮把 [prompt + 已接受 + 候选] 全量重新 prefill，
成本 O(L)——比普通 decode（O(1)）还贵，性能永远无法反转。原计划"接 APC 前缀缓存
加速 verify"，实现时发现 APC 只能救 prompt 前缀，救不了逐轮增长的已接受段；
真正的解法是**让 KV cache 轮间持久**（同一序列不换块表），verify 只 forward
新 token——这是同一目标的更优实现（APC 是跨请求共享前缀，这里是同请求
跨轮持久，底层都是"位置→槽位映射稳定则 KV 可复用"）。

**轮次结构**（`layers/mtp.py`，greedy verify）：
```
不变量：进入每轮时 cache 含位置 0..L-2，最后确认 token（位置 L-1）未前向
1. draft:  n-gram 从历史找尾部匹配 → k 个候选（位置 L..L+k-1）
2. forward [x_{L-1}, d_0..d_{k-1}]（k+1 个 token 一次 varlen 前向，
   复用 chunked-prefill 的 "q 是 k 的尾段" 形态：cu_seqlens_k = L+k）
   logits[i] = 位置 L+i 的分布 → greedy[i] 验证 d_i，greedy[k] 是全接受 bonus
3. 提交:   接受 draft[:n_acc] + 修正 greedy[n_acc]（最后 token 不前向 → 回到不变量）
```
成本从 O(L) 降到 O(k+1)。引擎直接驱动 model_runner + block_manager（绕开
scheduler），与普通 greedy 共享同一份权重和 KV 池——投机/普通模式切换零成本。

**被拒候选的垃圾 KV 为什么不污染**：forward 会把 d_{n_acc+1}.. 的 KV 写进
"未来位置"槽位，这些槽位在下一轮写入前不会被读——prefill kernel 的 key 循环
界是 cu_seqlens_k（确认长度），decode 是 context_lens，都只扫确认前缀；
下一轮同位置写入直接覆盖。PagedAttention 的位置→槽位映射固定是天然红利，
不需要任何清理逻辑。

**实测**（Qwen3-Omni-30B，H20，48 token greedy，`tests/test_speculative.py`）：

| 请求 | 基线 | 投机 | 加速 | 接受率 | 输出一致 |
|---|---|---|---|---|---|
| 重复这句话三遍 | 1.4s | 0.9s | **1.55x** | 83% | ✓ 逐 token |
| 介绍一下你自己 | 3.6s | 3.7s | ~1.0x | 0% | ✓ 逐 token |
| 数数 1-12 | 3.9s | 5.2s* | 0.75x* | 2% | ✓ 逐 token |

*req0 首轮包含 M=5 形状的 cuBLAS/MoE 首次选核开销（req2 热身后同场景
投机轮 ~79ms ≈ decode 步 76ms）。

**两个关键读数**：
1. **5-token verify 前向 ≈ 1-token decode 步**（79 vs 76ms）：decode 步是
   launch 开销主导（MoE 48 个 expert GEMM + 28 层 kernel 串联），多 4 个
   token 的算力是白送的——这是投机的收益空间：接受率越高白送越多。
   83% 接受率 → 每轮 ~3.5 token/5-token 前向 → 1.55x。
2. **接受率完全由 draft 质量/格式决定**：数数 prompt 用空格（"1 2 3"）生成
   用逗号（"1, 2, 3"），n-gram 上下文里根本没有这个格式的后续 → 2% 接受。
   n-gram draft 只在"生成内容复读上下文"时有效（重复句 83%）——这正是
   v7.1 MTP draft 的动机：模型自己的预测才不挑格式。

**踩坑**：无新增——v7.0 的四个坑（死循环/logits off-by-one/脏状态）在重写
架构下自然消失（手动状态只有块表和"末 token 未前向"一个不变量）。

## v7.1-v7.3 规划

- **v7.1 MTP draft**：Qwen3-30B-A3B 的 MTP 层权重当 draft head（一次前向
  出多 token 预测，比 n-gram 的接受率高一个量级）；
- **v7.3 动态 k**：接受率的滑动窗口 → 自适应 draft 长度（与 v6 的 P3
  "统计驱动调度"一脉相承）。
