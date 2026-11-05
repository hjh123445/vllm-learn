# Bench与Profile命令及指标口径

2026-11-05。命令从仓库根目录执行，示例中的MODEL替换为Qwen3-Omni公开权重目录。当前推理路径是文本Thinker，不把这些命令描述为完整音视频端到端推理。

## 一、先区分三种正确性

- 计算正确性：profiling前后相同输入输出应一致；这与HF/30B数值对齐是不同问题。先前DP2TP2 BF16自由生成分叉尚未解决。
- 采集正确性：有CPU/ATen和模块标签，还必须实际有CUDA kernel/runtime；分布式还要逐rank检查。文件名cuda不证明采到了GPU。
- 测量正确性：吞吐/分位数来自关闭profiling的运行，明确请求时间边界和token计数；采集开销不能当成模型性能。

## 二、离线吞吐与尾延迟

```bash
python v9/bench.py offline --model "$MODEL" --output outputs/bench_dp2tp4 \
  --dp 2 --tp 4 --enable-expert-parallel \
  --workloads short64,short128,context1k8,context4k8,mixed64,arrival64 --repeats 3
```

默认关闭profiling，同时清除继承的trace/logits/activation诊断配置。保存inputs、config、results、correctness和逐rank显存文件。每次运行保留每请求output_ids、token到达时间和原始step时间。

results中output_tok_s=实际输出token总数/该次负载墙钟时间；初始化及单独warmup不计入。request_s=完成请求数/墙钟时间。ttft_s是请求提交到首个有效生成token被EngineCore观察到；e2e_s到最后生成token；tpot_s=(最后token−首token)/(输出数−1)；itl_s汇总相邻生成token的间隔。报告P50/P95/P99，单位秒。

分位数使用排序后的实际观测值 `round((n-1)*q/100)`，无插值。raw requests保留样本数；64个请求的P99接近最大值，不能据此声称稳定生产P99。多DP时间戳来自同机副本EngineCore，不是客户端HTTP时间。arrival64是按engine step注入的模拟动态负载，不是Poisson在线负载。

EP由DP×TP派生，不额外乘卡；DP2TP4是8卡EP8。PP2TP4也是8卡，但每个stage EP4。

## 三、离线Profile

```bash
python v9/bench.py offline --model "$MODEL" --output outputs/profile_pp2tp4 \
  --pp 2 --tp 4 --enable-expert-parallel --workloads mixed64 \
  --profile --profile-start-message 3 --profile-messages 16
```

profile只接受一个workload、自动1次，不启用ablation，不先跑请求warmup消耗采集窗口。模型初始化的显存测量warmup在安装采集器前完成。输出results标记profiled=true/benchmark_valid=false，吞吐和分位数设为null，原始时间仅作诊断。每worker按本地run消息计数启动窗口；不同stage边界可能错开，需要用batch ID找到公共区间。

保存每rank的cuda_rank、capture_rank、audit_rank与pipeline_rank文件。cuda_rank包含自动ATen、CUDA runtime、GPU kernel和module hook范围；pipeline_rank只有CPU forward/P2P调用区间，不可用来估算实际GPU通信时长。

```bash
python v9/bench.py trace-check --input-dir outputs/profile_pp2tp4/trace \
  --expected-ranks 8 --require-cuda --require-nccl

python v9/tests/merge_pipeline_trace.py --input-dir outputs/profile_pp2tp4/trace \
  --same-host --output outputs/profile_pp2tp4/trace.json.gz
```

trace-check写trace_validation.json，缺ATen、层标签、所要求的CUDA/NCCL活动或rank文件数不匹配时返回非零。NCCL按kernel名称识别；某些实现改变命名时应检查原始数据。merge保留各rank/CPU/GPU stream与关联ID；本工具只处理已确认同机同次运行的时钟。

audit_rank不等于完整性能分析：不能仅凭kernel数量判断通信重叠。要在同一GPU不同stream看计算与NCCL区间交集，并检查依赖和无profile吞吐。

## 四、HTTP在线服务与Bench

服务（普通性能测量模式）：

```bash
python v9/server.py --model "$MODEL" --port 8000 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --max-num-seqs 128 --max-num-batched-tokens 512 --gpu-mem-util 0.70 \
  --metrics-path outputs/online_server_requests.jsonl

python v9/bench.py serve --base-url http://127.0.0.1:8000/v1 \
  --requests 512 --concurrency 64 --max-tokens 32 --warmup 8 \
  --output outputs/http_c64
```

服务当前一个AsyncEngine/EngineCore，可TP/PP/EP；多DP HTTP未实现，离线DPFrontend是另一入口。server接收文本messages，暂不支持多模态请求。

客户端是closed-loop：最多concurrency个在途请求，完成后补下一个。它支持 `--inputs public_or_synthetic.json`，格式为messages对象列表或文本字符串列表；默认自生成文本。它不自动读取机器上的数据集。

输出summary.json和逐请求requests.json、warmup.json。记录成功/失败数、实际output_tokens、output_tok_s、request_s，HTTP E2E/首非空内容TTFT/内容TPOT/内容chunk间隔的P50/P95/P99。

服务通过 `stream_options.include_usage=true` 返回真实completion_tokens；客户端只用usage计token，缺usage则输出token吞吐为null，绝不用chunk数估算token数。role-only/空content chunk不计首内容TTFT。客户端chunk间隔不是token ITL：多个token可合成一个chunk，特殊token可没有文本。内容TPOT使用首末非空内容时间与usage token数，也不等于每token内核时间；服务端token发布时间另保存在server metrics。

吞吐分母是整组请求提交到全部返回的墙钟时间，包含HTTP、排队、tokenization、服务调度和网络。成功请求吞吐仍包含失败请求占用的总时间；失败单独列出，存在失败CLI返回非零。当前只支持closed-loop，未实现Poisson开环或SLO goodput。

## 五、HTTP Profile入口

```bash
python v9/server.py --model "$MODEL" --port 8000 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --profile --profile-dir outputs/http_profile \
  --profile-start-message 3 --profile-messages 16

python v9/bench.py serve --base-url http://127.0.0.1:8000/v1 \
  --requests 64 --concurrency 32 --warmup 0 --max-tokens 32 \
  --profile --output outputs/http_profile_load
```

server的--profile会真正选择独立观测实现并采集有限窗口；进程优雅退出后还会保存窗口元数据与CPU时间线。客户端--profile只把负载标为诊断，不能远程开启普通服务器的profiler。客户端还会读取本项目/health的profiled_run，服务端声明profile时即使忘写客户端--profile也不输出性能吞吐/分位数；对其他服务无法自动保证其profiler关闭。

目前没有/start_profile或/stop_profile动态控制API，开始消息号从服务器启动后的实际run计数，不按请求计数；需重启带--profile服务来采新窗口。长驻服务重复采集、跨host时钟同步、Nsight原生采集不包含在本CLI内。

## 六、CUDA13空trace兼容处理

本机实测的PyTorch2.9.1+cu130环境，连独立矩阵乘法torch.profiler也出现有CPU无CUDA的情况；直接CUPTI活动接口有数据，且在profiler停止前显式flush后GPU记录恢复。独立采集器对CUDA13.0默认在同步后调用已安装匹配版本库的cuptiActivityFlushAll，再停止profiler。它不覆盖系统库，不修改engine，也不会在无profile运行执行。

可用 `VLLM_LEARN_CUPTI_FLUSH=off` 关闭兼容处理，`on`显式启用，默认auto仅匹配CUDA13.0。最终是否成功以audit的gpu_kernels/cuda_verified为准，不能只看cuda_available。额外flush/synchronize属于采集开销，不纳入吞吐基准。
