# 运行时清理：验收与八卡轨迹

2026-11-06，172.28.30.56，8×A800。使用隔离部署和自生成8层、hidden16、4专家checkpoint；不是Qwen3-Omni-30B质量或吞吐认证。

## 先看哪些文件

* [timeline.html](tiny_pp2ep4/analysis/timeline.html)：按rank/stage查看计算概览。
* [trace.json.gz](tiny_pp2ep4/trace.json.gz)：用Perfetto打开，包含八卡CPU线程、ATen、CUDA stream、NCCL和外置模块标注；没有裁剪原始采集事件。
* [summary.json](tiny_pp2ep4/analysis/summary.json)：逐消息层覆盖、CPU/GPU热点和区间交集。各rank实际记录12条消息，覆盖prefill/decode；stage0为层0–3，stage1为层4–7。基础检查无错误/警告，不意味着已证明profiler绝无丢事件。
* `tiny_pp2ep4/trace/cuda_rank*.json.gz`：各rank原始事件，无损gzip压缩；`pipeline_rank*.json`仅CPU调用，不可替代CUDA记录。
* [source_manifest.json](source_manifest.json)：实际部署的Python源SHA256。测试后仅把loader中“本stage只构造自己的参数”的不准确注释改为“非本stage的decoder层使用占位符”；源码token核对见`source_verification.json`。

## 如何生成

精确环境、路径和命令保存在各`*.command.txt`及`tiny_pp2ep4/manifest.json`。工作负载：PP2×TP4、开启派生EP4、8请求、每条8输入token、最多生成4token、调度预算16、从消息1开始采集最多16条，实际工作负载共12条。

```bash
python v9/tools/profile_capture.py --model /path/to/tiny --output outputs/tiny_pp2ep4 \
  --tp 4 --pp 2 --enable-expert-parallel --requests 8 --prompt-tokens 8 \
  --max-tokens 4 --budget 16 --start-message 1 --messages 16 \
  --max-kv-blocks 16 --max-model-len 128 --max-num-seqs 16 --record-shapes
python v9/tools/profile_analyze.py --input-dir outputs/tiny_pp2ep4/trace \
  --output outputs/tiny_pp2ep4/analysis --expected-ranks 8 --layers 8 --require-nccl
python v9/tools/profile_merge.py --input-dir outputs/tiny_pp2ep4/trace \
  --same-host --output outputs/tiny_pp2ep4/trace.json.gz
```

所有rank均有ATen、CUDA和NCCL，观察到每卡两个CUDA stream。此次同卡NCCL/非NCCL kernel时间交集为0，不能宣称通信计算已重叠；跨stage计算交集和同卡通信重叠是两个指标。小模型profile扰动很大，不用其速率、空白或交集比例推断30B收益。

## 验证通过与失败

| 检查 | 结果 |
|---|---|
| 完整默认pytest | 109 passed，8 skipped，12 subtests passed |
| 显式NCCL模型/EP/worker回归 | 10 passed，1 failed |
| 清理前源码单独复跑上述失败 | 相同BF16 token差异，见`before_dp2tp2_ep4.log` |
| 八卡capture/analyze/merge | 通过，日志和逐步退出码见`results.json` |

失败为DP2×TP2启用/关闭EP的严格自由生成对照：第二请求从`[54,60,57,60]`变为`[54,60,54,60]`。清理前已存在，此轮未修复；不能用固定前缀logits容差通过替代这项失败，也没有放宽该断言。完整日志见`runtime_nccl.log`。

较早的本轮重复trace已移出仓库，备份在整理说明所列仓库外目录的`runtime_trace.tar.gz`，不与本目录最终源码快照混用。
