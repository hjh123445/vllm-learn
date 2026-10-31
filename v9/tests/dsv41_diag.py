import os, sys, time
sys.path.insert(0, "/tmp/deepseekv4.1-A100-custom-main")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DSV41_LAYER_COUNTS"] = "6,6,6,6,5,5,5,4"
os.environ["DSV41_EP_TRACE"] = "1"      # per-layer 设备时间戳
CKPT = "/data/ifly_jyhe23/dsv4_rt/DeepSeek-V4-Flash"
from dsv41.engine import Engine, GenParams
eng = Engine(CKPT, devices=[0,1,2,3,4,5,6,7], max_seq_len=8192,
             use_graphs=False, ep=True, ep_shards=[32]*8, max_seqs=2)
ids = eng.tok.encode("Hi")
params = GenParams(max_new_tokens=2, temperature=0.6)
t0 = time.time()
for _, piece in eng.generate(ids, params):
    pass
print(f"\n[gen] 2 tok in {time.time()-t0:.1f}s", flush=True)
