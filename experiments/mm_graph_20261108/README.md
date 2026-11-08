# 2026-11-08 CUDA Graph / ChartQA 实验复现

实现基线为 `b3bed861c04f99c759feafd31cad1bd8292ac3ce`，与发布目录 `release_receipt.json` 的 `commit` 相同。发布整理只增加证据、脚本和文档，不修改这份已验证的 v9 实现。

入口：[报告](../../docs/CUDA_Graph多模态最终验证报告.md)、[原理](../../docs/CUDA_Graph与多模态原理.md)、[副本与分支审计](../../docs/最终版本与服务器副本核对.md)、[全部交付](../../outputs/releases/mm_graph_20261108/README.md)。

## 环境与数据

实测为 Linux、8×A800 80GB、BF16、TP8+EP8、PP1/DP1，依赖版本见证据归档的 `environment.json`。这是固定环境的复现记录，不是自动安装/升级脚本。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VLLM_LEARN_CPU_THREADS=4
export MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct
export PARQUET=/path/to/ChartQA/test.parquet
python v9/tools/evaluate_multimodal.py --mode prepare --parquet "$PARQUET" \
  --samples 256 --seed 20261108 --output datasets/chartqa/subset
export MANIFEST="$PWD/datasets/chartqa/subset/manifest.json"
```

数据来源为 `HuggingFaceM4/ChartQA` 的 test split（2,500 题），下载真实 parquet 后抽取固定 256 题。发布归档保存原始 parquet SHA256、抽样顺序、问题/答案和每张图 SHA256；不捆绑权重和数据集图片。重新准备后应核对证据归档 `dataset/manifest.json`，仅抽样数相同不够。HF 路径还需要该环境的多模态依赖，原机器的补充依赖目录记录在原始脚本中。

## 独立复现质量与严格 Graph 检查

以下从仓库根目录执行，使用新的输出目录。Graph/eager 的比较必须显式使用相同 batch-invariant 数值策略；普通 eager BF16 不能充当这个隔离对照。

```bash
python -m pytest v9/tests -q
PP_TEST_BACKEND=nccl python -m pytest v9/tests/test_pipeline_worker_runtime.py -q

python v9/tools/evaluate_multimodal.py --mode hf --model "$MODEL" \
  --manifest "$MANIFEST" --samples 256 --max-pixels 262144 --output-tokens 64 \
  --output outputs/reproduce/quality_hf

VLLM_LEARN_BATCH_INVARIANT=1 python v9/tools/evaluate_multimodal.py \
  --mode offline --model "$MODEL" --manifest "$MANIFEST" --samples 256 \
  --quality --graph --concurrency 128 --repeats 1 --max-pixels 262144 \
  --output-tokens 64 --output outputs/reproduce/quality_graph

VLLM_LEARN_BATCH_INVARIANT=1 python v9/tools/evaluate_multimodal.py \
  --mode offline --model "$MODEL" --manifest "$MANIFEST" --samples 256 \
  --quality --concurrency 128 --repeats 1 --max-pixels 262144 \
  --output-tokens 64 --output outputs/reproduce/quality_eager

python v9/tools/validate_decode_graph.py --model "$MODEL" \
  --output outputs/reproduce/strict_long --max-batch 256 \
  --batches 1 8 9 16 17 32 33 64 65 127 128 129 255 256 \
  --max-model-len 2048 --lengths 1023 1024 1535 2044 --steps 4 \
  --seed 1949 --random-inputs --permute --mixed-lengths \
  --strict-batch-invariant --batch-invariant --graph-reserve-gib 2
```

短上下文检查将 `--lengths` 改成 `17 255 256 511`。Graph 检查逐项断言完整 logits、真实 KV 和 dummy KV 哨兵，不能用生成文本相同替代它。多卡生产测试的默认 BF16 DP2×TP2 反例预期仍会失败；详见审计，不要放宽断言。质量统计按 `index` 比较完整 `output_ids`，EOS 停止；负载矩阵则强制固定输出长度，两者用途不同。

## 原始完整矩阵脚本

`original/` 保留最终实验原脚本的字节，原样版本还在发布的 `v9_final_verified_source.zip` 中。它们是带原机器路径的实验记录，不能直接在这个子目录运行。复现完整 26 档矩阵时，把这些脚本复制到一份独立仓库副本的根目录，在脚本中替换 `MODEL`、`MANIFEST`、HF `runtime_deps` 和 `package_evidence.py` 的模型/依赖元数据路径。`progress.py` 也有原机器绝对根目录；修改后的脚本应另存新 SHA256，不能再称与原脚本逐字节相同。

推荐用 `remote_job.py` 记录状态，先跑发布附带的 `run_revision_validation.py`，再跑 suite 和 finalize。原脚本的状态文件名是接口的一部分：

```bash
python remote_job.py --state outputs/revision_validation_wire/job.json -- python run_revision_validation.py
python remote_job.py --state outputs/completion_suite/resume_job.json -- python run_completion_suite.py
python finalize_results.py
python build_final_report.py --evidence outputs --output outputs/report
python package_evidence.py
```

矩阵为在线/离线、文本/图片、128/256 并发、small/medium/large 三档，共 24 档 Graph，加 medium 128/256 的离线 eager 两档，共 26 档，每档 3 轮。`run_load_matrix.py` 启动在线服务并记录执行元数据；`audit_outputs.py` 将 SSE 请求关联到服务端 token，比较重复请求、轮次、并发、在线/离线和 Graph/eager。`finalize_results.py` 追加正常 EOS eager 对照、真实图片 Vision Tower 对照以及最终验收。复现使用全新目录，脚本会保护已有结果。

本次发布没有重跑 29,952 请求矩阵：生产源码 SHA256 未变，已有最终矩阵完整封存；发布前补跑默认 pytest 和显式多卡生产回归，其原始日志另外保存于发布目录的 `release_audit/`。
