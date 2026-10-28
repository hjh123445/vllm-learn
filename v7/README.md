# v7：投机解码（Speculative Decoding）

建立在 **v6**（多卡并行引擎）之上——版本叠加，目录 = v6 全部 + 投机解码组件。

## 小版本演进

| 小版本 | 主题 | 状态 | 独立文档 |
|---|---|---|---|
| v7.0 | n-gram draft + verify 循环（正确性验证） | ✓ | 本文 §实现记录 |
| v7.1 | MTP 层 draft（Qwen3 自带权重，零训练） | 待做 | — |
| v7.2 | APC 前缀缓存加速 verify（性能反转点） | 待做 | — |
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

## v7.1-v7.3 规划

- **v7.1 MTP draft**：Qwen3-30B-A3B 的 MTP 层权重当 draft head（一次前向
  出多 token 预测，比 n-gram 的接受率高一个量级）；
- **v7.2 APC 加速 verify**：前缀缓存命中后 verify 只算新位置——性能反转；
- **v7.3 动态 k**：接受率的滑动窗口 → 自适应 draft 长度（与 v6 的 P3
  "统计驱动调度"一脉相承）。
