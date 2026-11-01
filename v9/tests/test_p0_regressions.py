"""P0 修复的回归测试（无 GPU 也能跑的部分 + GPU 部分标注）。

覆盖本轮修复中最容易回归的逻辑：
  T1  v0 幻影 token：混合 max_tokens 的 batch，length 行按自己的 done_len 截断
  T2  v2 off-by-one：len(prompt) % block_size == 0 的跨块 decode 领块时机
  T3  v9 _Spec 切片协议：第二轮 chunk / 混合批 decode 行的全局下标翻译
  T4  v9 Scheduler 准入校验：超长请求入队期拒绝（不进 assert 崩溃路径）
  T5  Sequence.peek_next_id：预读-消费-回退语义
  T6  quantized：padded buffer 宽度一致性（in_dim 非 group 倍数）
  T7  pack/unpack 往返（含负数 nibble、极小值组的 scale 下溢保护）

用法（每测试独立子进程跑，模块名冲突同 benchmark 的老问题）：
  python v9/tests/test_p0_regressions.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def spawn(worker_src: str, cwd: str):
    r = subprocess.run([sys.executable, "-c", worker_src],
                       capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        return {"pass": False, "detail": r.stderr.strip().splitlines()[-1] if r.stderr else "no stderr"}
    return json.loads(r.stdout.strip().splitlines()[-1])


T1 = r'''
import json, torch
from engine.batch import Batch
from samplingParams import SamplingParams
prompts = [[1,2,3,4], [5,6,7], [8]]
sps = [SamplingParams(temperature=0.0, max_tokens=2),
       SamplingParams(temperature=0.0, max_tokens=5),
       SamplingParams(temperature=0.0, max_tokens=5)]
b = Batch(prompts, sps, eos_id=999, pad_id=0, max_len=64)
# 模拟静态批跑到整批 5 步（行 0 的 max_tokens=2 应在第 2 步就完成）
outs = None
while not b.is_finished():
    n = torch.tensor([11, 22, 33, 44, 55][:b.bsz])
    b.append(n)
outs = b.collect()
lens = [len(o["token_ids"]) for o in outs]
ok = lens == [2, 5, 5]            # 旧行为: [5,5,5]（幻影 token）
print(json.dumps({"pass": ok, "detail": f"per-row lens {lens} (期望 [2,5,5])"}))
assert ok
'''

T2 = r'''
import json
from engine.kv_cache import KVCacheManager

# 最小复现：block_size=4, prompt 长 4（% 4 == 0）——decode 第 1 步写位置 4
# 应领第 2 块。旧版传 cur_len(5) → 5%4!=0 → 不领 → block_table 补零成 block0。
kv = KVCacheManager.__new__(KVCacheManager)   # 绕开 __init__ 的显存分配，手工装账本
kv.block_size = 4
kv.free_blocks = list(range(10))
kv.tables = {0: [0]}          # 准入时 allocate(4) 给 1 块

cur_len = 5                   # prompt 4 + 已生成 1（decode 喂的 token 写位置 4）
ok = kv.ensure_append(0, cur_len - 1)     # 修复后传写位置 4 → 4%4==0 领块
n_blocks = len(kv.tables[0])
passed = ok and n_blocks == 2
# 补零改 assert 的验证：请求超额块数应 assert（不再静默补零指向 block 0）
try:
    kv.block_table(0, 3)
    zero_ok = False
except AssertionError:
    zero_ok = True
print(json.dumps({"pass": passed and zero_ok,
                  "detail": f"ensure_append ok={ok} blocks={n_blocks} (期望2), "
                            f"超额块 assert={zero_ok}"}))
assert passed and zero_ok
'''

T3 = r'''
import json
from engine.core import _Spec

# 模拟第二轮 chunk：全局 [256:512) 的 256 个 token 被装进消息
tokens = list(range(1000, 1000 + 256))
m = {"seq_id": 0, "is_prefill": True, "num_cached_tokens": 256,
     "num_scheduled_tokens": 256, "block_table": [7, 8, 9],
     "tokens": tokens, "chunk_start": 256, "block_size": 256,
     "last_token": tokens[-1], "seq_len": 600, "temperature": 0.0}
s = _Spec(m)
# prepare_prefill 会做 seq[start:end] = seq[256:512]
chunk = s[256:512]
ok_slice = chunk == tokens
# 混合批 decode 行：num_cached>0、只带 1 个 token，chunk_start=全局位置
m2 = dict(m, num_scheduled_tokens=1, tokens=[42], chunk_start=512,
          last_token=42, is_prefill=False)
s2 = _Spec(m2)
ok_decode = s2[512:513] == [42]
# 越界必须 raise（协议破坏的第一现场）
try:
    s[600:700]
    oob = False
except IndexError:
    oob = True
print(json.dumps({"pass": ok_slice and ok_decode and oob,
                  "detail": f"chunk={ok_slice} decode_row={ok_decode} oob_raise={oob}"}))
assert ok_slice and ok_decode and oob
'''

T4 = r'''
import json
from engine.scheduler import Scheduler
from engine.sequence import Sequence
from samplingParams import SamplingParams
from config import Config
import tempfile, os

# 构造一个不用 GPU 的最小 Config（绕开 AutoConfig 需要模型目录——手工建假目录）
# Config 需要 model 目录 + AutoConfig；这里用一个真实存在过的微型配置 json
d = tempfile.mkdtemp()
import json as _j
with open(os.path.join(d, "config.json"), "w") as f:
    _j.dump({"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"],
             "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
             "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 32,
             "moe_intermediate_size": 32, "num_experts": 4,
             "num_experts_per_tok": 2, "vocab_size": 100,
             "rms_norm_eps": 1e-6, "rope_theta": 10000,
             "max_position_embeddings": 4096, "max_model_len": 512}, f)
cfg = Config(d, max_model_len=512, num_kvcache_blocks=8, kvcache_block_size=256,
             max_num_batched_tokens=1024, max_num_seqs=8)
sch = Scheduler(cfg)
# 超长请求（prompt+max_tokens > max_model_len）应在入队期拒绝
try:
    sch.add(Sequence([1] * 500, SamplingParams(temperature=0.0, max_tokens=100)))
    rejected_len = False
except ValueError:
    rejected_len = True
# 超池请求（需 2 块 > 池 8 块？不——构造需 9 块：2400 token）
try:
    sch.add(Sequence([1] * 2400, SamplingParams(temperature=0.0, max_tokens=10)))
    rejected_pool = False
except ValueError:
    rejected_pool = True
print(json.dumps({"pass": rejected_len and rejected_pool,
                  "detail": f"len_reject={rejected_len} pool_reject={rejected_pool}"}))
assert rejected_len and rejected_pool
'''

T5 = r'''
import json
from itertools import count
from engine.sequence import Sequence
a = Sequence.peek_next_id()
b = Sequence([1, 2])
c = Sequence.peek_next_id()
ok = (a == b.seq_id) and (c == b.seq_id + 1) and Sequence._peeked is not None
d = Sequence([3])          # 消费 peek
ok = ok and d.seq_id == c and Sequence._peeked is None
e = Sequence.peek_next_id() # 再次 peek 稳定
f = Sequence([4])
ok = ok and e == f.seq_id
print(json.dumps({"pass": ok, "detail": f"peek/consume cycle ids {a},{b.seq_id},{d.seq_id},{f.seq_id}"}))
assert ok
'''

T6 = r'''
import json, torch
from layers.quantized import QuantizedLinear, rtn_quantize, dequantize
# in_dim=130 非 group(128) 倍数：buffer 必须按 padded_k 分配（P0 潜伏 bug）
ql = QuantizedLinear(130, 8, group=128)
w = torch.randn(8, 130) * 0.05
ql.weight_loader(None, w)
rec = dequantize(ql.w_packed, ql.w_scales, 128)
rel = ((rec - w) ** 2).mean() / (w ** 2).mean()
# x 也要 pad 到 130？forward 走 T<=64 kernel 需要 GPU——CPU 只验证装载一致性
ok = ql.w_packed.shape == (8, 128) and ql.w_scales.shape == (8, 2)
print(json.dumps({"pass": ok and rel < 0.2,
                  "detail": f"packed{tuple(ql.w_packed.shape)} scales{tuple(ql.w_scales.shape)} rel_mse={rel:.2e}"}))
assert ok and rel < 0.2
'''

T7 = r'''
import json, torch
from layers.quantized import rtn_quantize, dequantize
torch.manual_seed(0)
w = torch.randn(64, 256) * 0.03
p, s = rtn_quantize(w, 128)
rec = dequantize(p, s, 128)
rel = ((rec - w) ** 2).mean() / (w ** 2).mean()
# 极小值组：amax ~1e-8 的组 scale 不再下溢为 0（clamp 6e-5）
tiny = torch.full((4, 128), 1e-8)
tiny[0, 0] = 1e-7
p2, s2 = rtn_quantize(tiny, 128)
ok_tiny = bool((s2 > 0).all())
print(json.dumps({"pass": rel < 5e-2 and ok_tiny,
                  "detail": f"rel_mse={rel:.2e} tiny_scale_positive={ok_tiny}"}))
assert rel < 5e-2 and ok_tiny
'''

CASES = [
    ("T1 v0 幻影 token 截断", T1, "v0"),
    ("T2 v2 ensure_append 差一", T2, "v2"),
    ("T3 v9 _Spec 切片协议", T3, "v9"),
    ("T4 v9 准入校验", T4, "v9"),
    ("T5 Sequence.peek_next_id", T5, "v9"),
    ("T6 QuantizedLinear padded 宽度", T6, "v9"),
    ("T7 RTN scale 下溢保护", T7, "v9"),
]

ROOT = os.path.dirname(os.path.dirname(HERE))       # 仓库根


def main():
    results = []
    for name, src, ver in CASES:
        results.append((name, spawn(src, os.path.join(ROOT, ver))))
    n_pass = sum(1 for _, r in results if r.get("pass"))
    print(f"\n===== P0 regressions: {n_pass}/{len(results)} PASS =====")
    for name, r in results:
        print(f"  [{'PASS' if r.get('pass') else 'FAIL'}] {name}  {r.get('detail', '')}")
    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
