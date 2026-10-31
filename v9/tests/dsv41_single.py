import os, sys, time
sys.path.insert(0, "/tmp/deepseekv4.1-A100-custom-main")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 单卡 offload 模式(专家进主机内存)——149G 权重的唯一单卡形态
CKPT = "/data/ifly_jyhe23/dsv4_rt/DeepSeek-V4-Flash"
from dsv41.engine import Engine, GenParams
eng = Engine(CKPT, devices=[0], max_seq_len=4096, use_graphs=False,
             offload_experts="cpu", max_seqs=1)
ids = eng.tok.encode("Hi")
t0 = time.time()
n = 0
for _, piece in eng.generate(ids, GenParams(max_new_tokens=4, temperature=0.6)):
    n += 1
    print(piece, end="", flush=True)
print(f"\n[gen] {n} tok in {time.time()-t0:.1f}s", flush=True)
