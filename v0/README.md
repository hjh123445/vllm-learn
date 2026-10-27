# v0：静态批处理

一切的地基。模型实现（RMSNorm / RoPE / GQA 注意力 / SwiGLU）全部手写、不依赖 transformers，
batch 组好就跑到整批完成。它有两个故意保留的缺陷——气泡和显存盲区——正是 v1/v2 要解决的。

## 快速开始

```bash
cd v0 && python example.py        # 4 条 prompt 一次进出
```

## 代码结构

```
engine/
├── batch.py        Batch：一个静态批的全部状态（序列缓冲区 + 生成进度）
├── scheduler.py    Scheduler：waiting 队列，schedule() 按 max_batch_size 组 Batch 出
├── llm_engine.py   Engine：生成主循环（唯一拥有循环的层）
└── model_runner.py ModelRunner：前向 + attention mask + 采样（纯计算）
layers/             RMSNorm / RoPE / 注意力 / SwiGLU / 采样，全部纯 PyTorch
models/qwen3.py     Qwen3（28 层 / hidden 1024 / GQA 2:1 / tie embedding）
utils/loader.py     safetensors 权重加载：名字映射（gate/up→fc1 融合、down→fc2）+ 装载对账
```

分层职责：**Batch 只管状态（get/append/is_finished），Scheduler 只管排队，
Engine 只管循环，ModelRunner 只管计算**。

## 主要技术原理与实现

### 静态批处理

Batch 组好就跑到整批完成，**slot 不回收**。短请求完成后 GPU 继续为它算 logits 和整层前向，
直到最慢的请求结束——这段浪费就是"气泡"（v1 的存在理由），用 `benchmark/bubble.py` 可视化实测 35%。

### Batch 的序列缓冲区（`engine/batch.py`）

一次性预分配 `[b, max_prompt_len + max_tokens]`：左边放 prompt（**左 padding**），
右边逐步写入生成的 token。prefill 和 decode 都从同一块 buffer 取视图，
positions 全长预计算（第 i 行第 j 列 = `j - pad_lens[i]`，pad 区 clamp 0）。
接口是两段式的：`get()` 返回本步的 `(input_ids, positions)`，`append(next_tokens)` 记账。

### 左 padding 与加性 attention mask（`engine/model_runner.py::_build_attn_mask`）

左 padding 让所有 prompt 右对齐，`logits[:, -1]` 对每条序列都是最后真实 token。
pad token 的 k/v 也在 cache 里，所以 softmax 前要挡掉：mask 是**浮点加性**的
（可见=0 不可见=-inf），规则一条：**causal = query 的 slot 号 ≥ key 的 slot 号，∧ 非 pad**。
这一条规则 prefill/decode 共用，没有两套逻辑。

### 权重装载对账（`utils/loader.py`）

HF checkpoint 的 MLP 是 gate/up/down 三个矩阵，我们是融合的 fc1（gate|up 拼接）+ fc2，
映射靠翻译表；装载后**对账**：`model_keys - loaded - allowed_missing` 必须为空，
这是抓住"漏装了还是 torch.empty 垃圾值"的唯一保险。tie embedding 在这里处理。

## 正确性与性能

- fp32 下与 HF transformers **逐 token 100% 对齐**（`benchmark/accuracy.py` T4）
- 12 请求混合负载：TPOT 17.9ms（Qwen3-0.6B）/ 25.2ms（Qwen3-8B）
- 完整修复记录见 `WORK_REPORT.md`（v0 开发期抓出的 11 个致命 bug）

## 已知限制（有意的）

- 气泡：整批最慢决定一切（v1 修）
- 显存按 `batch_size × max_seq_len` 拍脑袋预分配，实测利用率 7%（v2 修）
