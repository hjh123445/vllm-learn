# 工程问题实录：踩坑、排查与解决（面试向）

> 本文档记录项目全程真实遇到的问题。每条按 **现象 → 排查 → 根因 → 解决 → 原理/教训**
> 组织——面试官追问的正是排查过程和背后的原理，不是"我修了"。
> 按主题分组；⭐ = 高频面试题。

---

## 一、数值正确性问题

### 1.1 ⭐ RMSNorm 漏加 eps + fp32 原地操作毁掉残差

**现象**：bf16 下输出正常；fp32 下与 HF 对齐 0/32（全错乱），top1 token 都不对。

**排查**：bf16/fp32 行为不一致 → 逐层 hidden states 二分定位 → 第 0 层输出就发散（diff=11.9）
→ 层内逐子模块对比：norm/proj/q_norm 全部逐位一致后锁定 RMSNorm → 手算发现两个 bug 叠加。

**根因**（两个独立 bug）：
1. `rsqrt(x.pow(2).mean(-1))` 漏了 `+ eps`。Qwen3 的 embedding 数值幅度小，
   均方和 eps 同量级，误差显著；
2. **`x = x.float()` 在输入已是 fp32 时返回同一个张量（不拷贝）**，后续
   `x.mul_(...)` 原地改写 → 调用方的 **residual 被毁**。

**为什么 bf16 下没暴露**：`.float()` 从 bf16 升 fp32 会真拷贝，原地操作写的是副本。

**解决**：全 out-of-place 写法 `x * torch.rsqrt(mean + eps)`。

**教训**：dtype 转换方法"是否返回视图"是 API 的隐式契约；**测试矩阵必须包含 fp32**
（bf16 噪声掩盖了两个真 bug 几十个版本）。

### 1.2 ⭐ MoE 路由的 bf16 敏感性（v6）

**现象**：MoE 层 fp32 单元对拍 diff=0.0，但端到端 bf16 前缀一致率只有 1/3
（req1 2/2 过，req0/req2 在 token 10-32 处分歧）。

**排查**：先怀疑 mrope（Omni 特有的位置编码）→ 单独构造 HF 的 rope 模块对拍
→ inv_freq diff 1.9e-9、cos 差 0.002 < bf16 分辨率 0.0039 → **数学等价排除**
→ 剩下唯一对噪声敏感的点：router 的 topk 边界。

**根因**：router logits 的 bf16 噪声在 top-8 边界翻转专家选择（第 8/9 名的
概率差小于噪声量级），选错一个专家 = 该 token 的 MLP 输出完全不同 × 48 层放大。

**解决**：router 内部强制 fp32（softmax/topk/normalize 全 fp32 再转回）——
这在 HF 实现里本来就有（`dtype=torch.float`），是对齐后发现的深层原因。

**教训**：MoE 比 dense 对精度敏感一个量级；"逐位对齐单元 + 端到端统计对齐"
的双层验证策略是必要的（只做端到端会误判为"架构理解错误"）。

### 1.3 bf16 batch 形状差异的 argmax 翻转（v0-v3）

**现象**：同一请求 batch 跑 vs 单独跑，greedy 输出在 23-48 token 后分歧；
fp32 下完全一致。

**根因**：不同 batch 形状触发不同的 cuBLAS kernel（归约顺序/分块不同），
bf16 累加顺序差异 → logits 差 ~0.1 → 概率接近的 token argmax 翻转。

**解决**：fp32 做正确性测试（T2/T4），bf16 的这种分歧记为已知数值噪声。

**教训**：这不是 bug 而是浮点本质；**测试要知道自己测的是什么层面的正确性**。

---

### 1.4 ⭐⭐ DSV4 乱码三连：yarn 方向反 → 压缩器缺失 → 双 rope 体系（v6.3）

**现象**：DeepSeek-V4-Flash e2e 输出乱码 `'2 Programming# Ir...'`（管线跑通但数值全错）。

**排查**（三轮，每轮一个独立 bug，靠"输出前缀对了几 token"递进定位）：
1. **纸上推演 rope**：pos=5 处我们 cos=[0.999, 1.000, ...]（没旋转），HF cos=[0.284, -0.821, ...]——位置信息归零；
2. 修完 rope 后前 2 token 对、之后死循环 `10177`——数 token 窗口发现 21 层 CSA（rate=4）在 T≥4 就产压缩条目而我们没实现；顺手发现 `mla_attention` **压根没有 causal mask**；
3. 修完压缩器后仍循环——读 HF 源码发现 `position_embeddings` 是 **dict{"main", "compress"}**，41/43 层用 theta=160000 而非 10000。

**根因**（三个叠加）：
1. yarn 的 NTK-aware 插值是**频率侧**（inv_freq 混合 ramp），我们写成了**位置侧**（pos/factor）；correction range 的旋转圈数反解还把高频/低频方向弄反；
2. CSA 压缩条目（每 4 token 一条，Ca/Cb overlap 布局）未接入，21 层每层少一个 KV 条目；
3. HF config 的 `rope_parameters` 是 main/compress 双键 dict——sliding 层 default(theta=1e4)，CSA/HCA 层 yarn(theta=1.6e5) 且 `attention_factor=1.0` **显式覆盖** mscale 公式（我们多乘了 27.7% 幅度）。

**解决**：HF 原版 yarn 数学重写（对拍 diff=0）；CSA/HCA 压缩器 + causal mask 接入；双 rope 按层类型注入。最终 `' Paris. The capital of France is Paris...'` 语法连贯。

**教训**：
- **"config 打印 null 会骗人"**：`json.load(config.json)` 看到 `rope_parameters: null`，要用 HF 类读才知道它被 standardize 成双键 dict——读配置永远走模型类，不读原始 json；
- **量化模型的数值排查先看"位置编码有没有转起来"**（cos 幅度是最便宜的探针）；
- "前 N token 对然后崩"是层间误差累积的经典形状：N 越大说明剩余 bug 越靠后/越小。

### 1.5 LSE 合并的约定混用（v8.2 DCP）

**现象**：TP2×DCP2 四卡输出从 tok0 就是乱码，但单进程数学对拍全对。

**排查**：写单进程复刻（合成 KV、两个"假 rank"手工合并、不经 NCCL）→ NaN → 逐中间量检查发现空行的 `exp(-inf - (-inf)) = NaN` → 修掉 NaN 后 diff=2.99（固定值）→ 意识到不是数值噪声是**公式错**。

**根因**：LSE 合并有两套自洽约定——(A) parts 已归一化 + 权重含 l_i；(B) parts 未归一化 + 分母含 l_i。实现时 A 的公式配了 B 的 parts（v8.0 的 `lse_merge_stack` 是 A，`_dcp_attention` 手写时是 B 的产出）。单独看每边都对，合起来系数差一个 l_i。

**解决**：统一约定 B（`out = Σ e·part / Σ e·l`，e=exp(m_i-m*)），单进程对拍 3.6e-07 后再进引擎。

**教训**：**带数学的分布式组件必须有"单进程复刻"测试**（无通信、合成数据、双份实现交叉验证）——它把"通信问题还是数学问题"的二分成本从一次 15 分钟的 4 卡装载降到 30 秒。

## 二、GPU 显存问题

### 2.0 ⭐⭐ narrow().contiguous() 的视图陷阱：切分后显存纹丝不动（v6 TP/EP）

**现象**：EP 切分（每卡只留 64/128 个专家）后显存不降——61.1GB 纹丝不动；
TP 切分同样症状（当时靠提高 util 绕过没深究）。

**排查**（EP 时终于逼出根因）：
1. `sum(p.numel() for p in model.parameters())` 只有 32GB，但
   `torch.cuda.memory_allocated()` 61GB——**29GB 幽灵分配不在任何 Parameter 里**；
2. `torch.cuda.memory_snapshot()` 显示大量 0.81GB 的 active block
   （= 旧 w13 的整块尺寸 [128,1536,2048]×2B）；
3. gc.get_referrers 找不到 Python 引用——**引用在 C 层**；
4. 最小复现：`m.w = Parameter(m.w.data[:500].contiguous())` 后 allocated 不降；
   换 `.clone()` 立即降 2MB——**复现成功，根因锁定**。

**根因**：**`narrow()` 沿 dim0 的切片本来就是内存连续的，`.contiguous()` 发现
无需整理直接返回 view（共享 storage）**。"新"Parameter 与旧的全量权重共享同一块
显存——切分在逻辑上完成、物理上什么都没省。

**解决**：切分点全部 `.contiguous()` → `.clone()`（clone 无条件拷贝）。
修复后 EP 切分实测 61.1GB → 32.1GB。

**原理/教训**：
- PyTorch 的视图语义（view/share storage）是性能优化默认行为，**"这个调用
  会不会拷贝"不能靠方法名直觉，要靠 `data_ptr()` 对比或隔离测试**；
- 同一根因在两个版本各踩一次（TP 时绕过、EP 时才逼出）——**绕过的坑会回来
  找你**，第一次就该用最小复现定位；
- 这是并行切分代码的第一坑：vLLM/Megatron 用分片加载（每 rank 只读自己的
  shard）从根上避免整量加载再切。

### 2.1 ⭐ KV cache 预分配的隐式批广播（v0/v1）

**现象**：`IndexError: list index out of range`，报错位置在生成循环，
但单请求跑正常、batch>1 才炸。

**排查**：打印中间张量形状 → attention 的 logits 形状是 `[8, vocab]`
而 batch 只有 1 → q `[1,h,s,L]` 与 k `[8,h,L,hd]` 相乘没有报错而是**广播成了 8 行**。

**根因**：cache 按 `batch_size=8` 预分配，实际 batch=1 时 `get()` 返回 8 行，
PyTorch matmul 对 batch 维自动广播——**错误的数据静默产生形状合法的结果**。

**解决**：cache 记录 `cur_batch`，append/get 按实际 batch 切片。

**教训**：形状广播是最阴的 bug 类型（不报错、结果看似合理）；预分配的
维度必须在使用点重新对账。

### 2.2 MoE 朴素实现的显存爆炸（v6）

**现象**：`CUDA out of memory: Tried to allocate 768GB`。

**根因**：最初用 `einsum("th,tkh->tk", xs, w13[sorted_expert])`——
`w13[sorted_expert]` 把 128 个专家的权重**按 token 复制**（每 token 拷一份
3MB 的专家矩阵，T×k 个 token = 数百 GB）。

**解决**：改为按专家分组循环——每专家一次 `[n_e, H] @ [H, 2I]` gemm，
激活只含命中该专家的 token（显存 = 命中数 × 中间维度）。

**教训**：**gather 专家权重是 MoE 实现的第一大坑**（vLLM 用 fused grouped
gemm kernel 从根上避免）；先想清楚每个中间张量的实际尺寸再写。

### 2.3 8B 模型的 warmup 峰值挤爆池预算

**现象**：Qwen3-8B 在 `gpu_memory_utilization=0.3` 下断言 `num_kvcache_blocks > 0`。

**排查**：给断言加上各分量诊断（其他进程/激活峰值/权重）→ 打印出
"其他进程 5.3GB + warmup 峰值 22.8GB"——**0.3×85GB=25.5GB 的预算被吃满**。

**根因**：warmup 的 dummy 前向是 max_num_batched_tokens=4096 的满 prefill，
8B 的激活峰值大；加上共享 GPU 上其他进程的占用。

**解决**：8B 用 util 0.5；池预算断言改成带全部分量的错误信息（此后每次
环境问题它都能直接报出占用者——这个断言后来"自己证明了价值"三次）。

**教训**：**断言的报错信息就是未来的排查工具**；多用户共享 GPU 上任何显存
预算都必须实测当前空闲。

---

### 2.4 ⭐ warmup 临时池把 KV 预算吃掉一块：view 持有导致的"显存幽灵"（v3-v9 全系）

**现象**：KV 池总比理论预算小一截（v9.1 实测 372 blocks，理论 ~436，少 17%）。

**排查**（纸面审计，用户提问触发）：warmup 的 `tmp` 池通过 `_assign_layer_caches(tmp)`
把 view 挂在 attention 模块上——**函数返回后 tmp 并不释放**（view 持有 base storage），
要到 allocate 末尾换真池时才释放。但 allocate 开头量 `used = total - free` 时 tmp 还活着：
`budget = util×total − used − (peak−current) = util×total − W − T − S − A`，
T（临时池）被当成常驻显存扣了，稳态里它根本不存在 → 池子恰好小 `T/block_bytes` 块。

**两个连带发现**：
1. `peak − current` 这一项反而是**对的**（T 在 peak 和 current 里各一次，差值正好抵消）——
   所以修法不是"别扣 T"而是"量 used 前把 tmp 释放掉"；
2. v6-v9 的分支路由 bug：`if self.kv_cache is None` 单卡路径也是 True（warmup 不设置它），
   所以单卡走的也是 6GB 硬编码分支，docstring 声称的"warmup 实测峰值"是死代码。

**修复**（三步）：warmup 末尾记录 `A = peak − current`（此刻两者都含 T，差值正确隔离 A）→
显式打破模块 view + empty_cache 真正释放 tmp → allocate 用 `util×total − used − A`
（此时 used 只含 权重+context+其他进程）。

**陷阱提示**：如果只在 allocate 里给预算"+T 回加"而不释放 tmp，物理上 kv_cache 分配
瞬间 W+T+池 同驻，util 高时可能真 OOM——先释放再量才是安全的时序。

**教训**：
- **`narrow/索引 view 会延长父张量生命周期**"——与 2.0 的 narrow().contiguous() 是
  同族问题（那次是"不释放"，这次是"晚释放"），凡把 view 存进模块属性都要问一句
  "它几时死"；
- 显存预算公式的每一项都要标注**量测时刻的张量存活性**——`used`/`peak`/`current`
  三个数在不同时刻语义不同，纸面审计（W/T/S/A 记号推演）就能抓出这类错。

## 三、CUDA / Kernel 问题

### 3.1 ⭐ 逐序列朴素 attention 的 kernel 启动风暴（v2 重构时）

**现象**：重构为"每序列独立注意力"后 TPOT 从 39ms 涨到 101ms（2.6 倍退化）。

**排查**：profile 显示 Self CPU 6s vs Self CUDA 0.76s——GPU 87% 空闲。
数 kernel 数量：8 序列 × 28 层 × 每序列 5-6 个小 kernel = **每步 224+ 次启动**。

**根因**：0.6B 模型 decode 本来就是 CPU 发射瓶颈；kernel 数量翻几倍直接打爆。

**解决**：按形状分组——decode 组（全部 s=1 天然同形）合批成一条 `[n,h,1,L]`
批注意力（一次 gather + 一次计算）；prefill 行长度各异保持逐行（非热路径）。
TPOT 回到 36.6ms。

**教训**："长度不同"的正确处理是**同形合批 + 异形单独 + mask**，不是 padding
也不是逐个算；热路径和非热路径要用不同策略。

### 3.2 triton kernel 三个坑（v4）

1. **int32 溢出**：cache 偏移 `block_id × block_size × kvh × hd`，
   13000 block × 26 万元素 > 2^31——**静默写坏显存**（不报错，输出乱码）。
   解决：偏移一律 `.to(tl.int64)`。
2. **`tl.dot` fp32 默认 tf32**：fp32 对齐测试挂。解决：`input_precision="ieee"`。
3. **cu_seqlens 是绝对前缀和**：键长必须 `k[seq+1]-k[seq]` 相减；直接取
   `[seq+1]` 时第一条序列碰巧对（起点 0），第二条起全错——**单序列单测过了
   多序列才炸**，逐用例二分定位。

**教训**：kernel 的单测必须覆盖"第一条碰巧正确"的陷阱（多实例用例）；
int64 是 KV cache 索引的默认选择。

### 3.3 torch.compile 的 guard 税（nano-vllm 对比时发现）

**现象**：我们的 eager 引擎比 nano-vllm 的 eager 模式快（36.6 vs 48.2ms），
但 nano 开 CUDA graph 后 4.1ms。

**排查**：nano 的 profiler 按 **Self CPU 排序**（平时大家只看 aten 算子行），
浮出 `TorchDynamo Cache Lookup` 37,910 次 / 631ms + CompiledFxGraph dispatch
~2.5s——占其 CPU 的 40%。调用次数算术验证：28 层 × 2 个 compiled RMSNorm ×
步数 = 精确吻合 → **每次调用被编译函数都有固定 guard 开销**。

**根因**：nano 把 RMSNorm/sampler 都 `@torch.compile`；eager 模式下每次调用
都要查 Dynamo 缓存 + 评估 guard。decode 场景每步 57 次 × ~100µs = 数 ms。

**解决**（我们的 v4 策略）：**per-layer 的手写 triton（零 guard 税），
per-step 的（采样器）才用 torch.compile**。

**教训**：框架优化的收益要按调用次数算账；"看起来高级"的用法在错误的
粒度上是负优化。**profiler 换排序键**是发现这类问题的钥匙。

---

## 四、调度与架构问题

### 4.1 ⭐ 并发语义的口径陷阱（vLLM 对比时）

**现象**：与 vLLM 对比压测，我们 TTFT 20ms 完胜 vLLM 的 3 秒——数字"好到不信"。

**排查**：两边都读源码对语义 → nano 系 scheduler（我们 v3-v5 沿用）的
`max_num_seqs` 只限**每步批大小**，128 请求全部进 running（所以 TTFT 低、
TPOT 高）；vLLM 的 `max_num_seqs` 是**总并发硬顶**（其余排队，TTFT 高、TPOT 低）。

**解决**：给 vLLM 配 128 并发重测才可比。

**教训**：**跨系统对比前先对齐每个指标的定义**；"数字反常地好"本身就是
排查信号。这个发现也是项目定位的一部分（吞吐优先 vs 时延优先是服务策略选择）。

### 4.2 静态批处理的气泡（v0 → v1 的立项依据）

**现象**：v0 的 Gantt 图显示 35% 的 slot 时间在空转（bubble 可视化实测）。

**根因**：batch 组好后跑到整批完成，短请求完成后 slot 不回收，GPU 继续
为已完成的请求算整层前向。

**解决**：v1 的迭代级调度（每迭代准入/驱逐）——气泡归零。

**教训**：先可视化再优化；v0 是故意保留这个缺陷作为教学版本。

### 4.3 退出路径的信号处理缺失（v5 在线服务）

**现象**：`kill`（SIGTERM）服务进程后显存有时不释放（另一进程视角 19.9GB 占用）。

**排查**：两件事叠加——(a) 只有 Ctrl-C 走 finally 清理，SIGTERM 直接击杀
不走任何清理；(b) **我的启动脚本 `cd x && nohup python ... &` 的 `$!` 是
bash 包装进程的 pid**，kill 打在包装上 python 变孤儿（这个坑我自己踩了三次）。

**解决**：注册 SIGTERM/SIGINT handler → engine.stop()（join 线程）→
srv.shutdown()（**必须从别的线程调**，signal handler 跑在 serve_forever 同一线程，
直接调会死锁）→ server_close；启动脚本用 `exec nohup` 或 ps 拿真实 pid。

**教训**：优雅退出是服务的必备路径不是锦上添花；signal handler 的线程语义
（跑在主线程）决定了清理代码不能直接调阻塞 API。

---

## 五、分布式与多节点（v6 进行中）

### 5.1 节点 56 的 home 目录满导致 SSH 免密配不上

**现象**：paramiko 推公钥显示成功，`ssh` 仍 Permission denied；
SFTP 写文件报 `OSError: Failure`、shell echo 重定向静默失败。

**排查**：ssh-copy-id 的输出里翻到 `cat: write error: No space left on device`
——**节点 56 的 home 磁盘 100% 满**。sshd 无法写自己的日志/临时文件。

**解决**：公钥写到 `/data`（两节点共享存储）再 `ln -sf` 到 `~/.ssh/authorized_keys`。

**教训**：跨机器配置失败时先查目标机的磁盘/权限基本健康；
共享存储可以作为满盘机器的配置通道。

### 5.2 模型下载的路径选择

**现象**：huggingface_hub 下载 61GB 模型，几小时只下了 2MB。

**排查**：逐层测——代理对大文件 503/reset（curl 直连测出 hf-mirror 只有
175KB/s）、ModelScope 直连 4.75MB/s（快 27 倍）→ 但 61GB 仍要 3.6h。

**解决**：放弃下载，盘点本机已有模型（DeepSeek-V4-Flash 149GB +
Qwen3-Omni-30B 66GB，都满足需求）。

**教训**：下载前先测实际带宽；**"已有的资源"优先于"理想的资源"**。

---

### 5.3 ⭐⭐ NCCL 多 communicator 的 bootstrap 竞态：25 分钟死锁（v8.2）

**现象**：TP2×DCP2 四卡，layer0 的第一个 cp 组 all_gather 永久挂起；某 rank 报 `ncclRemoteError: socketPollConnect: connect 172.29.226.143:57039 returned Connection refused`（重试 35 次）。

**排查**：
1. 最小复现（裸 `new_group` × 2 + 双组 all_reduce 预热，无任何引擎代码）**同样死锁** → 排除业务代码；
2. `faulthandler.dump_traceback_later` 在挂起 150s 后自动转储全线程栈 → **铁证**：两个进程栈在 `tp_allreduce`、两个在 `all_gather`——交叉等待；
3. 对比实验：`init_process_group` 加 `device_id` 参数 → 立即全通。

**根因**：`new_group` 的 NCCL communicator 是**惰性 wire-up**（首次 collective 才建立），多组并发初始化时 TCP bootstrap 有竞态；不绑 `device_id` 时设备亲和未定，竞态从偶发变成稳定复现。

**解决**：两件套——`init_process_group("nccl", device_id=...)` + `warmup_groups`（每维组启动时做一次小 all_reduce 强制立即完成 wire-up，之后运行期无惰性初始化）。

**教训**：
- **多 communicator 是雷区但不是禁区**：混合并行（TP 组 + cp 组 + world 组）必须预热，且所有 rank 以相同顺序预热（DIMS 顺序）；
- `faulthandler.dump_traceback_later(timeout, exit=True)` 是死锁排查的第一工具——比 py-spy 轻（无需权限）、比 gdb 快（自动触发）、比"再加打印重跑 15 分钟装载"便宜两个数量级；
- "connection refused" 在 NCCL 语境下常常不是网络问题，是**对端进程还卡在上一个 collective**（bootstrap socket 还没监听）。

### 5.4 git 走代理推 GitHub 的 gnutls 失败 → Git Data API 兜底

**现象**：`git push`（含 `-c http.proxy`）一律 `gnutls_handshake() failed: The TLS connection was non-properly terminated`，但 `curl` 走同一代理访问 api.github.com 正常。

**解决**：写推送脚本走 **Git Data API**（blob base64 → tree(base_tree) → commit → PATCH ref）——内容一致但 SHA 与本地不同（API 重建提交）。脚本每次以远端 HEAD 为 base，天然兼容这种分叉。

**教训**：Git 的 HTTP 传输层（git 的 gnutls）和系统 curl 的 TLS 栈是两套实现，一个坏一个好很常见；Data API 是完整的提交图操作接口，推送只是它的 1% 能力。

## 六、基础设施问题

### 6.1 公司代理的 git 推送（贯穿全程）

**现象**：git push 频繁 `gnutls_handshake failed` / `CONNECT tunnel 503`。

**排查**：三层原因——(a) `~/.gitconfig` 里给 github.com 配的旧代理端口已死；
(b) 出口代理本身间歇 503；(c) API 域名通而 git 域名被卡。

**解决**（递进的三套方案）：
1. `-c http.https://github.com.proxy=<新端口>` 临时覆盖；
2. 503 时重试 + 退避；
3. 仍失败走 **GitHub Git Data API 手动构造提交**（blob→tree→commit→PATCH ref，
   以远端 tree 为基只传 diff 文件）——绕过 git 协议完全走 HTTPS API。

**教训**：CI/推流通道要有降级路径；Git Data API 是 git 协议不可用时的
完整替代（连删除文件都能表达：tree entry 的 sha 传 null）。

### 6.2 vLLM 与主环境的 torch 版本冲突

**现象**：想装 vLLM 做对照，但 `pip install vllm` 要把 torch 2.9.1 顶成 2.13。

**解决**：专用 venv 统一装（vllm + 项目依赖一起，vllm 带全 torch/triton/
transformers，只需再补 xxhash）；实测我们的引擎在 torch 2.13 下全部跑通
（accuracy 5/5、TPOT 16.2ms）。

**教训**：锁版本的依赖（vLLM 锁 torch）要隔离；但"隔离"不等于"两套环境"——
**一个 venv 同时装两边**比维护两个环境好（对照实验同环境才算公平）。

### 6.3 压测的 SYN 风暴误判

**现象**：64+ 请求同时发起，出现 Connection reset；稳态压力下 0 错误。

**根因**：stdlib http.server 的 accept 队列默认 5，瞬时建连风暴打爆它
（不是引擎问题）。

**解决**：`request_queue_size=256` + 压测用到达间隔（20ms）而非全突发；
文档里明确"建连风暴 ≠ 稳态压力"。

**教训**：压测工具本身要区分"瞬时并发"和"稳态负载"，两者测的是不同的东西。

---

## 七、方法论沉淀（面试常问"你怎么发现 X 的"）

| 问题类型 | 发现手段 |
|---|---|
| 数值错误 | **fp32 测试矩阵**（bf16 噪声掩盖真 bug）；逐层 hidden 二分 |
| 形状/广播错误 | 逐模块打印 shape；"单请求过 batch 炸"提示批维度问题 |
| kernel 级错误 | 逐 kernel 对拍纯 PyTorch 参考（fp32 err ~1e-7 才算过） |
| 性能归因 | profiler **换排序键**（Self CPU vs Self CUDA）；调用次数与模型结构做算术验证 |
| 跨系统对比 | 先对齐指标定义/口径，"数字反常地好"就是排查信号 |
| 显存问题 | `memory_allocated/max_memory_allocated` 直接读账本；断言带全分量诊断 |
| 环境问题 | 好的报错信息是排查工具（池预算断言三次自己报出占用者） |
