# v4：手写 triton kernel 层

v3 的调度栈 + **全部手写的 kernel 层**：融合算子（残差+RMSNorm、q/k norm+rope+cache 写入、
SiLU×up）+ 手写 triton FlashAttention（paged decode 在线 softmax / paged prefill causal）。
torch.compile 只用在每步一次的采样器上。

## 快速开始

```bash
cd v4 && python example.py
```

## 代码结构

```
layers/
├── kernels.py      手写融合算子（见下表）
├── flash_attn.py   手写 FlashAttention：decode 在线 softmax / prefill causal（tl.dot）
└── attention.py    薄封装：读 Context → 调融合 kernel + flash kernel
engine/sampler.py   torch.compile 的采样器（每步一次，guard 税可忽略）
models/qwen3.py     残差管道 vLLM 化（每层的加法融进下一个 rmsnorm kernel）
```

## 主要技术原理与实现

### 为什么 per-layer 手写 triton、per-step 才用 torch.compile

torch.compile 的 guard 税按**调用次数**收：nano-vllm 把 RMSNorm/sampler 都 @torch.compile，
eager 模式实测每步 57 次 dynamo guard 查找吃掉 40% CPU（PROFILE_NOTES §7.1）。
所以：**每层每步都跑的 → 手写 triton（零 guard 税）；每步只跑一次的 → torch.compile**。

### 融合算子清单（`layers/kernels.py`，每个都逐一对拍过纯 PyTorch 参考）

| kernel | 融合了什么 | 替代的 aten ops |
|---|---|---|
| `fused_add_rmsnorm` | 残差加法 + RMSNorm（残差原地更新） | 7 |
| `silu_and_mul` | SwiGLU 后半 | 4 |
| `fused_qk_norm_rope_store` | q norm + k norm + rope(q,k) + **cache 写入** | 30+ |

第三颗最狠：每个 token 一个 program，q/k 逐头 rmsnorm、rope，然后 k/v **直接从寄存器
散射进 cache 槽位**——不再经过"先写显存再 index_put"的往返。残差管道相应 vLLM 化：
每层的残差加法由下一层的 `fused_add_rmsnorm` 消费，**全模型 forward 零 aten add**。

### 手写 FlashAttention（`layers/flash_attn.py`）

- **decode**（`_decode_attn_kernel`，grid=(n, num_heads)）：每 program 一个 (序列, q头)
  流式扫自己的 block 表，**在线 softmax**（m/l/acc 滚动更新），K/V 直读 cache——
  零 gather 零 repeat_interleave，GQA 头组映射在 kernel 内（`h // (HQ/HK)`）。
- **prefill**（`_prefill_attn_kernel`，grid=(seqs, q_blocks, heads)）：causal **底右对齐**
  （query 绝对位置 = seqlen_k − seqlen_q + j，chunked prefill / 前缀缓存通用），
  `tl.dot` 走 tensor core。

数值约定：load 后升 fp32 计算；fp32 输入下与纯 PyTorch 参考**逐位一致**（~1e-7），
这是 fp32 HF 对齐测试的前提——所以 `tl.dot` 必须用 `input_precision="ieee"`（默认 tf32 会挂）。

## 踩坑（都在 kernel 单测里抓的）

1. `cu_seqlens_k` 是**绝对前缀和**：键长必须 `k[seq+1] − k[seq]` 相减——直接取 `[seq+1]`
   时第一条序列碰巧对（起点为 0），第二条起全错；
2. cache 偏移一律 **int64**：13000 block × 26 万元素/block 溢出 int32 静默写坏显存；
3. `tl.dot` fp32 默认 tf32。

## 与 v3 的差异

| | v3 | v4 |
|---|---|---|
| TPOT | 39.0ms | **13.4ms**（2.9 倍） |
| 吞吐 | 156.4 | **405.6 tok/s** |
| TTFT avg | 117 | **23.7 ms**（prefill 单 kernel 化） |
| kernel 数/层 | ~25 aten | ~10 triton |

对 vLLM 的位置：eager 吞吐 1800 vs vLLM eager 1602 tok/s（同量级）；vLLM graph 模式
5551 的差距全部来自 CUDA graph（v4 的固定形状 Context 正是捕获前提）。

## 正确性

T4 fp32 与 HF **100% 逐 token 对齐**。v4 全程 fp32 计算比 bf16 torch 路径**更精确**，
所以 bf16 跨引擎逐位等价（T3）对 v4 不适用（bf16 argmax 深处分歧 = 数值噪声，实测 token 23-48/48）。
