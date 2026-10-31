"""dsv41 运行时 smoke: V4-Flash 8×A800 EP 模式装载 + 生成。

参考 YJH008/deepseek-v4.1-flash-a800-deploy 的 8 卡配方:
  --ep --ep-shards 32,32,32,32,32,32,32,32  (256 experts ÷ 8)
  DSV41_LAYER_COUNTS=6,6,6,6,5,5,5,4       (43 层 ÷ 8, 管道序连续)
差异: V4-Flash 是 43 层/256 专家(V4.1 是 40/384), shard 数相应调整。
"""
import os, sys, time
sys.path.insert(0, "/tmp/deepseekv4.1-A100-custom-main")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DSV41_LAYER_COUNTS"] = "6,6,6,6,5,5,5,4"
os.environ.setdefault("DSV41_NVCC", "/usr/local/cuda/bin/nvcc")
CKPT = "/data/ifly_jyhe23/dsv4_rt/DeepSeek-V4-Flash"

import torch
from dsv41.engine import Engine, GenParams

t0 = time.time()
eng = Engine(CKPT, devices=[0, 1, 2, 3, 4, 5, 6, 7], max_seq_len=8192,
             use_graphs=False, ep=True,
             ep_shards=[32] * 8, max_seqs=2)
print(f"[load] {time.time()-t0:.0f}s", flush=True)

ids = eng.tok.encode("The capital of France is")
params = GenParams(max_new_tokens=32, temperature=0.6)
t0 = time.time()
n = 0
for _, piece in eng.generate(ids, params):
    n += 1
    print(piece, end="", flush=True)
dt = time.time() - t0
print(f"\n[gen] {n} tokens in {dt:.1f}s = {n/dt:.1f} tok/s", flush=True)
