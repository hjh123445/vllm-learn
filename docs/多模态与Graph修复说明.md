# 多模态、观测隔离与 CUDA Graph：本轮实现边界

> 2026-11-08 最终更新：本文保留历史排查过程。最新修正、768 组真实图像特征对拍、256 题 Graph/eager 检查及 128/256 并发在线/离线总表见 [最终验证报告](CUDA_Graph多模态最终验证报告.md) 和 [原理说明](CUDA_Graph与多模态原理.md)。

2026-11-07。batch 补齐的完整调查、严格模式、实验和复现见 [CUDA_Graph补齐精度排查.md](CUDA_Graph补齐精度排查.md)。本文补充同轮其它修改和旧性能数字的核对。

## 1. 观测隔离已恢复

删除 models/qwen3_moe.py 的 VLLM_LOGIT_DUMP、CPU 拷贝和 torch.save；还发现并删除 layers/attention_reference.py 的 SDPA capture 落盘逻辑。正常 compute_logits 只调用 lm_head。采集集中在独立 observed_runner、module hooks 和诊断工具里，设置旧环境变量也不会让模型写盘。

## 2. grouped GEMM 的 +8.9% 是哪个实验

原始证据来自此前 tuning/base_argsort 与 tuning/final_grouped，不能与旧简历中 +89% 混用。环境为同节点8×A800-SXM4-80GB，Qwen3-Omni-30B-A3B Thinker 文本、BF16、TP8、DP1、EP8、PP1，48层、128专家、top8，hidden2048、expert intermediate768。offline EngineCore、eager、greedy、ignoreEOS；每配置只测2轮。

初始一次性64请求 burst，不是在线稳定并发64。token budget512，max-seqs128，microbatch128，max-len8192，显存利用率0.7，关闭prefix cache；正式计时前8请求×4输出token预热。初始化和tokenizer不计，prefill/decode/调度IPC计入。profile单独采集。

|负载|实际输入/输出|baseline输出tok/s均值|grouped输出tok/s均值|变化|
|---|---|---:|---:|---:|
|short64|10请求47输入+54请求48输入；每请求32输出，共3062输入/2048输出|97.0455|100.2629|+3.32%|
|mixed64|48短47–48输入+16长2047–2048输入；42请求16输出+22请求64输出，共35062输入/2080输出|52.6302|57.3023|+8.88%|

mixed64 的输入+输出总token速率是939.8026→1023.2323，不能说生成速度1023 tok/s。两负载首轮优化前后完整生成均64/64一致，只证明这次变更的生成对拍，不等于HF parity。输入SHA256相同；记录的4个源码指纹中moe.py不同、其它3个相同，但并非完整源码冻结、随机交错、多重复统计实验。两轮结果不支持统计显著性或所有场景+8.9%的承诺。

本轮未复跑该历史吞吐；核验依据另保存在 work/decode_graph_review_20261107 的原始JSON和 evidence_summary.json，并将相应 tuning 输入/结果一并放入本轮本地证据目录。

## 3. 图片理解分支实际做了什么

当前实现为图片输入→Thinker文本输出；并非音频、视频、Talker全Omni。需要模型的vision权重、processor与图像解码依赖、像素和grid元数据、图像占位token跨度、三路mRoPE位置以及语言侧KV管理。

数据流：processor产生patch pixels、grid_thw、token ids及位置→完整图片过vision encoder/merger→merged特征替换语言embedding相应行，前三层加入deepstack特征→按chunk切语言prefill→常规decode使用已缓存KV。不是把图片直接转成普通整数token就结束了。

修复：旧实现按语言chunk切原始patch，却保留整图grid，破坏vision二维位置与attention。现在先编码完整图片，再按语言chunk切 merged 和3路deepstack。engine/vision_cache.py 有界LRU按请求/图片缓存；最后视觉行消费后释放，超容量可重新编码。decode IPC只发mRoPE delta，不重复像素和完整pos3。多模态禁止按相同image_pad token IDs复用文本APC，防止两张不同图片错误共享KV。

入队校验MM开关、pos3形状[3,L]、span合法且不重叠，暂只支持PP1。显存预算显式预留encoder/cache/Graph空间；固定预留不能保证任意分辨率不OOM。图片特征缓存减少重复vision计算，但尚无受控端到端加速结论。

验证：真实vision权重、自生成不同尺寸图片，与HF逐项比较 merged+3路deepstack，FP32和保留FP32 inv_freq的BF16参考均max_abs=0；chunk特征拼接一致、各图编码一次、最终缓存归零。首次参考模型全量to(BF16)把频率buffer也转低精度，出现差异，排查后统一频率策略；失败未隐藏。TP2/EP2双图片、chunk32和完整prefill均能生成，但文本并未逐位一致，完整多模态HF质量未通过，不能把路径跑通说成质量验证完毕。

## 4. Graph 的真实修复

旧支持检查把所有带backend属性的模块都要求triton，GroupedExperts当然不满足，导致旧所谓Graph实验可能全程eager；现在按模块类型判断，显式请求不支持时报原因，保存captured_buckets/replays等执行证据。

动态all-to-allv split不能直接捕获整图。本实验dense EP路径利用DP1内TP复制输入，对本地全部专家做固定形状计算，再合并贡献；这会计算未激活专家，FLOPs约为稀疏top-k的E/k=16倍，不是通用最优算法。A800原native grouped_mm捕获会内部复制offsets而失败，静态路径使用BMM；严格模式改用固定tile Triton GEMM。

修复路由布尔索引的动态shape、专用dummy KV块/slot=-1、inference_mode捕获与replay、异常context恢复、同stream数据填充和replay、退出前释放Graph。真正发生replay的八卡CLI trace已验证。图内没有重新执行Python layer hooks，不能虚构逐层标注覆盖。

旧Graph的+12%不能继续当本实现的实测收益：历史版本Graph未实际运行，且dense/QK变体有生成差异。本轮完成正确性与可观测性排查，没有新的受控吞吐结论。后续性能比较必须固定同一数值策略、输入、调度参数，分开benchmark和profile。
