#!/bin/sh
export PYTHONPATH=/tmp/vllm-learn-ep-coupled-20261105-205927/testdeps:v9
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for name in final_bf16_tp final_bf16_ep final_precise_tp final_precise_ep; do
  case "$name" in
    final_bf16_tp) options='' ;;
    final_bf16_ep) options='--ep' ;;
    final_precise_tp) options='--expert-precision float32' ;;
    final_precise_ep) options='--ep --expert-precision float32' ;;
  esac
  timeout --signal=TERM --kill-after=10 240 /home/ifly_jyhe23/env/miniconda3/bin/python v9/tools/diagnose_ep.py --model tiny --output "$name" $options > "$name.log" 2>&1
  status=$?
  echo "$status" > "$name.exit"
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
echo 0 > verified_queue.exit
