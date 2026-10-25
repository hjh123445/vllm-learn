# vllm_learn v0 静态批处理推理框架 — 修正工作报告

日期：2026-10-23
对象：`vllm_learn/`（Qwen3-0.6B，无 transformers 的手搓推理路径 + HF tokenizer）
目标：`python example.py` 跑通；多请求静态批处理输出正确；batch 内外结果一致。

---

## 一、验证结果（修复后的验收）

| 验证项 | 结果 |
|---|---|
| `python example.py`（单请求，chat template，temperature=0.6） | 通过，输出通顺的自我介绍 |
| 3 条不同长度 prompt 静态批处理（greedy） | 通过，各条输出切题，无串扰 |
| 同一 prompt 单独跑 vs 放 batch 里跑，greedy 32 token 逐 token 对比 | **完全一致**，证明左 padding / 位置 / mask 无串扰 |

---

## 二、文件改动明细

> 共修改 11 个文件、新增 1 个、删除 2 个。每条按"位置 → 问题 → 原理 → 修正"组织。

### 1. `layers/attention.py`（重写，本文件 bug 最密集）

**1.1 `Attention.__init__` — `self.num_kv_head = num_head` 赋值错**
- 位置：原第 33 行
- 问题：把 q 头数 16 存进了 `num_kv_head`，GQA（8 个 kv 头）被静默变成 MHA（16 个 kv 头）。
- 原理：GQA 里 `repeat_interleave(num_head // num_kv_head, dim=1)` 的复制比由这个字段决定；16/16=1 意味着 k/v 完全不复制，16 个 q 头各自只看 8 个 kv 头中的前若干个，注意力权重全错但不报任何错。
- 修正：`self.num_kv_head = num_kv_head`。

**1.2 `Attention.__init__` — 构造签名与调用方不匹配**
- 位置：原第 23-29 行
- 问题：签名是 `(batch_size, max_position, hidden_dim, num_head, num_kv_head, scale)`，没有 `head_dim`；`models/qwen3.py` 按关键字传 `head_dim=...` 会直接 TypeError。
- 原理：qwen3 的 `head_dim=128` 不等于 `hidden_size=1024 / num_heads`（Qwen3 是显式配置 head_dim 的架构），所以 head_dim 必须由外部传入而不能自己推。
- 修正：签名改为 `(batch_size, max_position, num_head, num_kv_head, head_dim, scale)`，删除 `hidden_dim` 推导逻辑。

**1.3 `Attention.forward` — 完全忽略 `attn_mask`**
- 位置：原第 39-48 行
- 问题：签名没有 `attn_mask`，调用方传了也会因参数不匹配报错；而静态批处理左 padding 必须屏蔽 pad 位置。
- 原理：左 padding 下 pad token 的 k/v 也会写进 cache。如果不在 softmax 前把 pad 位置的分数置成 `-inf`，模型会"读到"另一批 pad token 的内容，污染所有真实 token 的注意力分布。
- 修正：`forward(q, k, v, attn_mask)`，在 softmax 前加 `scores = scores + attn_mask`（浮点加性 mask，形状 `[b,1,s_q,cur_len]`，可广播到所有 q 头）。

**1.4 `KVCache` — 普通 tensor 属性不随模型迁移（致命）**
- 位置：原第 8-11 行 `self.k = torch.zeros(...)`
- 问题：KV cache 用普通属性存张量。`nn.Module.to(device, dtype)` 只搬 parameters 和 buffers，普通 tensor 属性留在 CPU、float32。
- 原理：权重全部在 GPU/bf16 上，attention 里 `q @ k` 一个在 CUDA 一个在 CPU 直接 RuntimeError；即使 device 碰巧对上，fp32 cache 和 bf16 权重的混合计算也会引入精度和显存问题。
- 修正：`self.register_buffer("k", ..., persistent=False)`。`persistent=False` 同时保证它不进 `state_dict`，不干扰 loader 的对账逻辑。

**1.5 `KVCache.get` — 预分配 batch 维不按实际 batch 切（隐蔽，最后一个修掉的 bug）**
- 位置：`KVCache.get`
- 问题：cache 按 `batch_size=8` 预分配，实际 batch=1 时 `get()` 返回 `[8, kv头, len, hd]`。
- 原理：这是本次修正里最值得记住的一课——**PyTorch 的 matmul 会广播 batch 维**。q 是 `[1, 16, s, 128]`、k 是 `[8, 8, L, 128]`，`q @ k.transpose(-2,-1)` 不报错，广播成 `[8, 16, s, L]`；logits 变成 `[8, vocab]`，`engine/llm.py` 里 `finished[i]` 越界报 IndexError。报错位置（生成循环）和病灶（cache 切片）隔了三层，是靠打印中间张量形状才定位到的。
- 修正：`append()` 记录 `self.cur_batch = k.shape[0]`，`get()` 按它切片 `self.k[:b, :, :cur_len, :]`。

**1.6 `KVCache.add` — `len` 遮蔽内建函数、无溢出检查**
- 位置：原第 13-17 行
- 问题：`len = k.shape[2]` 覆盖了内建 `len`；写入越界无检查（越界会静默写坏别处或报难懂的 slice 错误）。
- 修正：改名为 `n`，加 `assert cur_len + n <= 容量`，方法名 `add→append`、新增 `reset()`（`cur_len=0`，靠覆盖写新 batch）。

### 2. `layers/activation.py`（重写）

**2.1 方法名拼错 `forawrd`**
- 位置：原第 8 行
- 原理：Python 里方法名写错不是错误，`self.act(x)` 找不到 `forward` 才会 TypeError（`'SiluAndMul' object has no attribute 'forward'`）。nn.Module 的 `__call__` 走 `_call_impl` 查找名为 `forward` 的方法。
- 修正：改为 `forward`。

**2.2 `F.sliu` 不存在**
- 位置：原第 10 行
- 原理：torch.nn.functional 里函数名是 `silu`（Sigmoid Linear Unit）。拼错会在调用时 AttributeError。
- 修正：`F.silu(x) * y`。顺带把类名 `SliuAndMul` 改为 `SiluAndMul`，删除对这种一行前向无收益的 `@torch.compile`（v0 阶段它只带来编译启动开销和调试困难）。

### 3. `layers/norm.py`

**3.1 `RmsNorm.__init__` 漏 `super().__init__()`（致命）**
- 位置：原第 11 行（`self.weight = nn.Parameter(...)` 之前）
- 原理：`nn.Module` 的内部簿记（`_parameters`/`_modules`/`_buffers` 字典）在 `Module.__init__` 里建立。没调 `super().__init__()` 就给模块赋 `nn.Parameter`，`Module.__setattr__` 找不到 `_parameters`，直接 `AttributeError: cannot assign module before Module.__init__() call`。
- 修正：构造第一行加 `super().__init__()`。同时删除 `@torch.compile`（理由同上）。

### 4. `layers/sampler.py`（重写）

**4.1 `torch.max(y)[1]` 的索引方式错误**
- 位置：原第 6-8 行
- 原理：`softmax` 后取最大对 2D 输入 `[b, vocab]`，`torch.max(y)[1]` 混用了 `torch.max(y)` 返回 namedtuple 的取值方式；正确写法是 `torch.argmax(logits, dim=-1)`。
- 修正：greedy 分支直接 `argmax(logits, dim=-1)`。
- 原理补充：`argmax(softmax(x)) == argmax(x)`（softmax 单调），先 softmax 纯属白算，还引入数值下溢风险。
- 注：engine 最终采用的采样在 `model_runner.py` 的 `sample()`（含 temperature/top_p），本文件保留为独立 greedy/temperature 采样组件。

### 5. `layers/rotary_embedding.py`

**5.1 `use.sin` 少括号**
- 位置：原第 23 行 `torch.cat([use.cos(), use.sin], dim=-1)`
- 原理：`Tensor.sin` 不带括号是方法对象（builtin_function_or_method），不是值。构造时即 TypeError。
- 修正：`use.sin()`。

**5.2 RoPE 结果与 q/k 的广播维度错位（隐蔽、静默）**
- 位置：原第 25-27 行 forward
- 问题：`cos_sin_cache` 形状 `[max_pos, 1, head_dim]`，按 `positions [b, s]` 索引得 `[b, s, 1, head_dim]`；直接 chunk 后与 q 的 `[b, 头数, s, head_dim]` 相乘，`s` 维和头数维对不上，广播失败。
- 原理：q/k 的形状约定是 `[b, num_head, s, head_dim]`，位置相关的 cos/sin 必须是 `[b, 1, s, head_dim]`（头数维为 1 才能广播到所有头）。
- 修正：`self.cos_sin_cache[positions].transpose(1, 2)`。
- 另注：RoPE 本身用的是 half-split（chunk 成两半）而非 interleaved 风格，与 HF Qwen3 的 `rotate_half` 实现一致，这部分没动。

### 6. `models/qwen3.py`

**6.1 引用了 config 里不存在的 `config.max_seq_len`（当时）**
- 位置：原第 35 行 `max_position=config.max_seq_len`
- 问题：调用时刻直接 AttributeError。（后来 config.py 补上了 `max_seq_len` 字段，这一条随之解决。）
- 修正：构造链 `Qwen3ForCausalLM(config)` → `QwenModel(config)` → 各层统一读 `config.batch_size` / `config.max_seq_len`。

**6.2 RoPE 表 28 份重复（浪费显存）**
- 位置：原 `QwenAttention.__init__` 每层各建一份 `RotaryEmbeddding`
- 原理：RoPE 的 cos/sin 表只跟 (head_dim, max_position) 有关，全模型共享一份就够。28 层 × 约 20MB ≈ 570MB 纯浪费。
- 修正：`QwenModel.__init__` 建一份，循环注入每层 `layer.self_attn.rope = rope`（`persistent=False` 的 buffer 不会进 state_dict，共享同一模块实例没有对账问题）。

**6.3 KV cache 容量按 40960 预分配（约 10 倍浪费）**
- 位置：原第 34-35 行，`max_position=config.max_position_embeddings`（40960）
- 原理：cache 显存 = 层数 × 2 × kv头数 × head_dim × max_len × batch × 每元素字节。按 40960 算是 28×2×8×128×40960×8×2B ≈ 3.7GB；引擎实际只会用到 max_seq_len=4096（≈370MB）。
- 修正：容量统一用 `config.max_seq_len`。

### 7. `engine/model_runner.py`

**7.1 import 路径错误**
- 位置：原第 5 行 `from loader import load_model`
- 问题：loader 在 `utils/loader.py`，模块顶层搜不到 `loader`。
- 修正：`from utils.loader import load_model`。
- 另注：你后续重写了本文件（引擎参数全部收进 config、采样挪进 `ModelRunner.sample`），最终版保留你的结构，只改了这一行 import。

### 8. `utils/loader.py`

**8.1 `from glob import glob` 遮蔽后调用 `glob.glob`（致命）**
- 位置：原第 3 行 + 第 44 行
- 原理：`from glob import glob` 后，名字 `glob` 指向函数而不是模块，`glob.glob(...)` 即 `函数.glob` → `AttributeError: 'function' object has no attribute 'glob'`。
- 修正：`import glob`。

**8.2 权重映射表漏 `down_proj`（致命）**
- 位置：`PACKED_MAPPING` / `map_name`
- 问题：checkpoint 里 MLP 是 `gate_proj/up_proj/down_proj` 三个矩阵，映射表只处理了前两个（融合进 `fc1`），`down_proj` 原名直传，模型里叫 `fc2`，`model.get_parameter("...down_proj.weight")` 报 `QwenMLP has no attribute down_proj`。
- 原理：这个 loader 的设计是"名字对账制"——checkpoint 名 → 模型名的翻译表必须完备，否则对账环节之前就会先在 get_parameter 崩掉。
- 修正：新增 `RENAME_MAPPING = {"down_proj": "fc2"}`，`map_name` 先查融合表、再查改名表。

### 9. `engine/llm.py`（你写的主循环，修了两处）

**9.1 decode 阶段 attention mask 双重错误（致命，且是最烧脑的一个）**
- 位置：原 `build_mask(s_q, cur_len)` + decode 调用 `build_mask(1, self.runner.cur_len)`
- 问题一（causal 判断错）：`causal = torch.arange(s_q)[:, None] >= slots[None, :]`，s_q=1 时 `arange(1)=[0]`，即"query 的 slot 0 只能看到 cache 的 slot 0"——decode 步的新 token 实际位于 cache 的**最后一个** slot，应能看到之前所有 slot。按原实现，每步 decode 只注意第一个 token。
- 问题二（长度少 1）：调用时传的 `cur_len` 是 append 之前的值，而 `Attention.forward` 是先把当前 token 的 k/v 写进 cache 再做注意力，实际分数矩阵是 `[b, h, 1, cur_len+1]`，mask 是 `[b,1,1,cur_len]`，形状对不上。
- 修正：`build_mask(query_slots, cur_len)` 改为显式接收"各 query token 在 cache 里的 slot 号"；prefill 传 `list(range(max_prompt))`，decode 传 `[runner.cur_len]` 且总长传 `runner.cur_len + 1`。
- 原理：causal 的本质是"query 所在 slot 号 >= key 所在 slot 号"。把它写成显式的 slot 比较，prefill/decode 就统一成同一条规则，不需要两套逻辑。

**9.2 采样只用了 `sps[0]`（见第三节遗留问题 1）**
- 位置：`self.runner.sample(logits, sps[0].temperature, sps[0].top_p)`
- 问题：`max_tokens`/`ignore_eos` 是按请求用的，temperature/top_p 却整批共用第 0 条的。
- 修正：本次未动（记为遗留问题），已在循环里补上 `ignore_eos` 的按请求判断。

### 10. `samplingParams.py`（统一为唯一定义）

**10.1 装饰器错误（致命）**
- 位置：原第 4 行 `@dataclasses`
- 原理：`@dataclasses` 是把模块本身当装饰器用，类被替换成模块相关对象。正确写法是 `@dataclasses.dataclass`。
**10.2 assert 方向反了（致命）**
- 位置：原第 11 行 `assert self.temperature < 1e-10`
- 问题：断言"温度必须接近 0"，而默认值是 1.0——构造默认参数必炸。
- 修正：统一为 nano-vllm 风格的参数集（temperature/top_p/max_tokens/ignore_eos/seed 预留），校验改为 temperature>=0、max_tokens>0。
- 另注：原来 `engine/llm.py` 里还有第二份字段不同的 `SamplingParams` 定义，两处定义漂移是隐患，已统一到本文件，`engine/llm.py` 改为 import。

### 11. `llm.py`（顶层入口）

**11.1 `class LLM(engine)` 继承了模块（致命，第一次运行的报错点）**
- 位置：原第 1-3 行 `from engine import engine` / `class LLM(engine)`
- 原理：`engine` 是 package（目录），`from engine import engine` 拿到的是 `engine/engine.py` **模块对象**；继承模块会触发 `module() takes at most 2 arguments`。
- 修正：顶层 `llm.py` 变成 re-export：`from engine.llm import LLM, SamplingParams`（`example.py` 用的是 `from engine.llm import ...`，顶层文件保留给 `from llm import LLM` 的旧用法）。

### 12. 新增 `utils/tokenizer.py`

- 说明：框架原本没有任何 tokenize 能力。引擎当前用 `AutoTokenizer`（见第五节说明），本文件是基于 `tokenizers` 库（Rust，独立于 transformers）的替代实现：`tokenizer.json` 加载 + encode/decode + eos/pad 解析（generation_config.json 里 eos 可能是 list）。
- 用途：后续想彻底去掉 transformers 依赖时直接换入。

### 13. 删除 `engine/engine.py`、`engine/scheduler.py`

- 说明：修正中途我按早期 API 设计实现过一版 Engine/StaticBatch，你随后把主循环整合进了 `engine/llm.py`，两套并存会造成"改一处忘另一处"的漂移，删除以保持单一实现。

### 14. `config.py`（你自行补充，未再改动）

- `max_seq_len: int = 4096` 字段补上后，`Qwen3Config.from_pretrained` 的"字段名与 config.json 一致 + engine_kwargs 过滤"设计即可正常工作，该文件不再有改动。

---

## 三、遗留问题（不影响当前功能）

1. **采样参数未 per-request 化**：`engine/llm.py` 中 `runner.sample(logits, sps[0].temperature, sps[0].top_p)` 整批共用第 0 条请求的 temperature/top_p。修法：`ModelRunner.sample` 的输入从标量改为按行张量——temperature 变 `[b,1]` 与 logits 相除天然按行；top_p 核截断本来就是按行 sort/cumsum，天然支持每行不同阈值；greedy 行（temperature=0）先按 1 参与归一化、最后用 argmax 覆盖该行结果。
2. **tokenizer 依赖 transformers**：核心推理路径（模型、attention、RoPE、loader、采样）完全无 transformers；只有 `engine/llm.py` 和 `example.py` 的 tokenize 用了 `AutoTokenizer`（主要为了 chat template）。可用 `utils/tokenizer.py` + jinja2 渲染 chat template 替换。
3. **缺对齐单测**：建议加一个"与 HF transformers 同 checkpoint 逐 token 对比 logits（容差 bf16 下 1e-2~1e-3）"的单测，作为后续任何改动的回归基线。
4. **性能欠账（v0 预期内）**：attention 是朴素 SDPA（可换 `F.scaled_dot_product_attention`）、`build_mask`/`prefill_positions` 每步在 CPU 上用 Python 循环构造、`repeat_interleave` 每步复制 kv（可用 `enable_gqa` 或 expand 视图）。

---

## 四、验证环境

- 模型：`Qwen3-0.6B`（safetensors，tie_word_embeddings=true）
- 软件：torch 2.9.1+cu130 / safetensors 0.8.0 / tokenizers 0.22.2 / python 3.10
- 硬件：单卡 GPU（A 系列 80GB）
