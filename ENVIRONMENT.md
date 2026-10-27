# 环境说明（ENVIRONMENT.md）

本项目开发与测试的真实环境，以及**复现时踩过的所有环境坑**（每一坑都真实发生过）。

## 1. 硬件 / 系统

| 项 | 值 |
|---|---|
| GPU | A800-80G ×4（Ampere，**sm80**），单卡单进程 |
| GPU 共享 | **多用户共享**：常有其他进程占 4–25GB，测试全部显存预算都走 warmup 实测 |
| OS | Ubuntu 24.04，Linux 6.8 |
| Python | 3.10（miniconda） |

## 2. 主环境依赖（跑 v0–v5 + benchmark）

```
torch==2.9.1+cu130        # 2.4+ 即可；cuda toolkit 13.0（/usr/local/cuda-13.0，nvcc 可用）
triton==3.5.1             # v4 的手写 kernel 依赖
transformers              # 仅用于 tokenizer 和正确性对照（HF 参考）；5.x 注意事项见下
safetensors==0.8.0        # 权重加载
tokenizers==0.22.2        # utils/tokenizer.py 的无-transformers 加载
xxhash                    # v3+ 的 APC block 哈希
```

```bash
pip install torch triton transformers safetensors xxhash
```

**transformers 5.x 注意**：`rope_theta` 从 config 顶层挪进了 `config.rope_parameters`，
项目代码已兜底（直接读 config.json）。正确性对齐测试（T4）用 transformers 加载 HF 参考模型。

## 3. 模型

本地路径（不在仓库内），默认在仓库根上一级找，可用环境变量覆盖：

```bash
export VLLM_LEARN_MODEL=/path/to/Qwen3-0.6B
```

| 模型 | 规格 | 兼容性 |
|---|---|---|
| Qwen3-0.6B | 28 层 / hidden 1024 / GQA 2:1 / **tie embedding** | 六版全兼容 |
| Qwen3-8B | 36 层 / hidden 4096 / GQA 4:1 / **untied embedding** | 六版全兼容（loader 的 tie 分支 + kernel 的 `h // (HQ/HK)` 天然覆盖） |

**8B 的显存预算**：`gpu_memory_utilization` 至少 0.5（0.3 时 warmup 峰值
权重 16.4GB + 激活就吃满预算 → 池子不足 → decode 触发抢占风暴）。
0.6B 测试用 0.25–0.3。

## 4. flash-attn（**仅 nano-vllm 对照需要**，v0–v5 不依赖）

主环境没有匹配 torch 2.9+cu130 的预编译 wheel（官方 wheel 只到 cu12torch2.8），
需源码编译：

```bash
pip install ninja
# 源码：GitHub API 下 tarball（git clone 走代理常失败，见 §6）
# 关键：GitHub release 的 tarball 不含子模块，csrc/cutlass 会缺失（assert 报错），
# 需按 .gitmodules 记录的 commit 用 API 补齐 cutlass（composable_kernel/aiter 是 ROCm 的，不需要）
cd flash-attn
MAX_JOBS=32 CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="8.0" \
    python setup.py build_ext --inplace
# .so 出现在仓库根（flash_attn_2_cuda*.so），拷到 site-packages 顶层（flash_attn 包与其同级）
```

验证：`import flash_attn; flash_attn_varlen_func(...)` 能在 GPU 上跑通即 OK。

## 5. vLLM：统一环境（一个 venv 同时跑我们的引擎和 vllm）

**为什么不直接装进主环境**：vLLM 锁死 torch 版本（0.31 要求 2.13.0，实测会把主环境
torch 2.9.1 顶掉），连带后果：(a) 主环境按 torch 2.9 ABI 编译的 flash-attn 直接挂；
(b) base conda 是多项目共用的。所以正确做法是**一个专用 venv 统一装 vllm + 本项目依赖**，
我们的引擎和 vllm 对照在同一个环境里跑：

```bash
python -m venv /path/to/vllm-learn-unified     # 持久路径（别放 /tmp，重启就没了）
pip install vllm                               # 会带入 torch 2.13.0+cu130 / triton 3.7 / transformers 等
pip install xxhash                             # vllm 没带的只有这一个（其余项目依赖 vllm 都带全了）
# 验证统一环境：全引擎回归 + vllm serve 都用它
/path/to/venv/bin/python benchmark/accuracy.py --test 1        # 实测 5/5 PASS（torch 2.13 上）
/path/to/venv/bin/python benchmark/speed.py --engine v4       # 实测 TPOT 16.2ms（主环境 13.8，triton 3.7 vs 3.5 的 codegen 差异）
/path/to/venv/bin/vllm serve $VLLM_LEARN_MODEL --port 8030 \
    --gpu-memory-utilization 0.3 --max-model-len 4096 --max-num-seqs 128
```

**例外：nano-vllm 对照（benchmark/nano.py）需要 flash-attn**，它是按主环境 torch 2.9 编译的——
nano 对照在主环境跑（或对统一环境的 torch 2.13 重编译一遍 flash-attn，约 25 分钟，流程见 §4）。

**坑**：`kill` 父进程后 **EngineCore 子进程仍占着显存**，必须 `pkill -f "vllm serve"`
清场，否则下次启动报 "Free memory less than desired"。
**坑**：`pkill -f "vllm serve"` 会匹配到包含该字符串的自身命令行（自灭），
写脚本时用变量拼接规避。

## 6. 网络（公司代理，坑最多的部分）

出口代理：`http://172.30.212.55:3128`（环境变量 http(s)_proxy 已全局设置）。

| 坑 | 现象 | 解法 |
|---|---|---|
| git 全局配置的代理过期 | `~/.gitconfig` 里给 github.com 配了 `:7890`（已死）→ `gnutls_handshake() failed` | push/pull 时 `git -c http.https://github.com.proxy=http://172.30.212.55:3128 ...` 临时覆盖（别改全局配置） |
| 代理间歇性 503 | `CONNECT tunnel failed, response 503` | 重试几次；仍失败用 GitHub **Git Data API** 推送（blob→tree→commit→ref，以远端 tree 为基只传 diff 文件；项目里已用通整套流程） |
| git clone github 走代理不稳 | clone 挂 | 用 GitHub API 下 tarball（`/repos/X/tarball/refs/heads/main`）替代 |

## 7. 复现 benchmark 的注意事项

- 正确性（accuracy.py）要 fp32 加载双份模型，**GPU 上别有其他大任务**，
  否则池预算断言会直接把占用者报出来（断言信息含"其他进程 xx GB"——这是特性）。
- 服务级压测（serve_bench.py）：**20ms 到达间隔的稳态满压 0 错误**；
  64+ 连接**瞬时 SYN 风暴**会打爆 stdlib http.server 的 accept 队列（已调 256 但仍可能 reset），
  不要用 `--arrive-interval 0` 的全突发当有效数据点。
- 对比 vLLM 时注意**并发口径**：我们的 `max_num_seqs` 只限每步批大小不限总并发（吞吐优先），
  vLLM 的是总并发硬顶（时延优先）——两边要配成同一口径（见 PERFORMANCE_NOTES §9）。

## 8. 进程管理（自己踩了三次的坑）

`cd x && nohup python ... &` **整条链路被后台化**，`$!` 拿到的是 bash 包装进程的 pid，
kill 打在包装上、python 变成孤儿继续占 GPU。正确姿势：

```bash
exec nohup python server.py ... &    # exec 替换掉包装 shell
# 或事后 ps 拿真实 pid:
ps aux | grep "[s]erver.py"
```

验证进程真死透：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`。
