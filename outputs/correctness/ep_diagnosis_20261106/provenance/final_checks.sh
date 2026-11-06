#!/bin/sh
export PYTHONPATH=/tmp/vllm-learn-ep-coupled-20261105-205927/testdeps:v9
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PP_TEST_BACKEND=nccl timeout --signal=TERM --kill-after=10 240 /home/ifly_jyhe23/env/miniconda3/bin/python -m pytest v9/tests/test_pipeline_worker_runtime.py::test_production_dp_ep_unequal_batches_and_idle_replicas -q -s > default_regression.log 2>&1
echo $? > default_regression.exit
timeout --signal=TERM --kill-after=10 120 /home/ifly_jyhe23/env/miniconda3/bin/python -m pytest v9/tests/test_route_reduction.py v9/tests/test_silu_and_mul.py v9/tests/test_ep_token_ownership.py v9/tests/test_ep_topology.py -q > unit_regression.log 2>&1
echo $? > unit_regression.exit
/home/ifly_jyhe23/env/miniconda3/bin/python v9/tools/compare_logits.py --cached-reference bf16_tp_v3 --cached-actual bf16_ep --output bf16_comparison.json > bf16_comparison.log 2>&1
echo $? > bf16_comparison.exit
/home/ifly_jyhe23/env/miniconda3/bin/python v9/tools/compare_logits.py --cached-reference precise_tp --cached-actual precise_ep --output precise_comparison.json > precise_comparison.log 2>&1
echo $? > precise_comparison.exit
/home/ifly_jyhe23/env/miniconda3/bin/python v9/tools/compare_logits.py --cached-reference fp32_tp --cached-actual fp32_ep --output fp32_comparison.json > fp32_comparison.log 2>&1
echo $? > fp32_comparison.exit
