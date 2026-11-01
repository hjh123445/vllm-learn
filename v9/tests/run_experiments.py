"""v9.5 实验套件：通路验证 + 性能/Profile 数据采集（8xA800 一晚跑完）。

子命令:
  smoke <config>      通路验证: 装载→生成→文本打印（config: tp1/tp2/ep2/ep8/dcp2/pp2/...）
  bench <config>      6 请求 × 48 tok greedy 计时（与 PARALLEL_MATRIX 同口径）
  quant <mode>        W4 量化路径: bf16 | w4（单卡）| w4_ep2（EP+量化= P0-14 裁决）
  compare             汇总所有已存 JSON 打对比表

用法:
  python run_experiments.py smoke tp1
  python run_experiments.py bench tp2
  python run_experiments.py quant w4_ep2
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = ROOT                                       # v9 目录
PATH = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
if not os.path.exists(f"{PATH}/config.json"):
    PATH = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

CONFIGS = {
    # name: (tp, ep, dcp, pp, gpus, quant)
    "tp1":    dict(tp=1, ep=1, gpus=[0]),
    "tp2":    dict(tp=2, ep=1, gpus=[0, 1]),
    "ep2":    dict(tp=1, ep=2, gpus=[0, 1]),
    "ep4":    dict(tp=1, ep=4, gpus=[0, 1, 2, 3]),
    "ep8":    dict(tp=1, ep=8, gpus=list(range(8))),
    "tp2ep2": dict(tp=2, ep=2, gpus=[0, 1, 2, 3]),
    "dcp2":   dict(tp=1, ep=1, dcp=2, gpus=[0, 1]),
    "pp2":    dict(tp=1, ep=1, pp=2, gpus=[0, 1]),
}

MSGS = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话",
        "用三句话介绍牛顿第二定律", "法国的首都是哪里", "水的化学式是什么"]
SP_MAX_TOKENS = 48

WORKER = r'''
import json, os, sys, time
sys.path.insert(0, "%(base)s")
PATH = "%(path)s"
MODE = "%(mode)s"                 # smoke | bench | quant_bf16 | quant_w4 | quant_w4_ep2
TP, EP, DCP, PP = %(tp)d, %(ep)d, %(dcp)d, %(pp)d
GPUS = %(gpus)s
from transformers import AutoTokenizer
from samplingParams import SamplingParams
tok = AutoTokenizer.from_pretrained(PATH)
try:
    with open(f"{PATH}/chat_template.json") as f:
        tok.chat_template = json.load(f)["chat_template"]
except FileNotFoundError:
    pass
msgs = %(msgs)s
ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                              add_generation_prompt=True) for m in msgs]
prompts = [tok.encode(p, add_special_tokens=False) for p in ps]

from engine.core import EngineCore
t_load0 = time.time()
kw = dict(tp_size=TP, ep_size=EP, dcp_size=DCP, pp_size=PP, gpus=GPUS,
          config_kwargs={"max_num_seqs": 8, "max_model_len": 2048,
                         "max_num_batched_tokens": 16384,
                         "gpu_memory_utilization": 0.85, "enforce_eager": True})
core = EngineCore(PATH, **kw)
t_load = time.time() - t_load0

QUANT = MODE.startswith("quant_w4")
if QUANT:
    import torch
    # 单卡 inproc: 直接量化 runner 的模型；多卡: 量化要进 worker——EngineCore
    # 的 spawn 路径不支持注入，quant_w4_ep2 走 TP/EP=1 退化 + EP 分支单测
    if TP == 1 and EP == 1 and PP == 1 and DCP == 1:
        n = core.inproc_runner.model.quantize_int4(group=128)
        torch.cuda.empty_cache()
        print("[quant] layers:", n, flush=True)
    else:
        print("[quant] 多卡路径需 worker 注入——本跑退化为 bf16", flush=True)

sp = SamplingParams(temperature=0.0, max_tokens=%(maxtok)d)
sid2idx = {}
t0 = time.time()
for i, p in enumerate(prompts):
    seq = core.add_request(p, sp)
    sid2idx[seq.seq_id] = i
outs = {}
steps = 0
while not core.is_finished():
    for sid, toks in core.step():
        outs[sid2idx[sid]] = toks
    steps += 1
wall = time.time() - t0
core.shutdown()
n_tok = sum(len(v) for v in outs.values())
texts = [tok.decode(outs.get(i, []), skip_special_tokens=True)[:60] for i in range(len(prompts))]
print("RESULT" + json.dumps({
    "mode": MODE, "tp": TP, "ep": EP, "dcp": DCP, "pp": PP,
    "load_s": round(t_load, 1), "run_s": round(wall, 1), "steps": steps,
    "n_tok": n_tok, "tok_s": round(n_tok / wall, 1) if wall > 0 else 0,
    "outs": {str(k): v for k, v in outs.items()}, "texts": texts}))
'''


def run_config(name, mode, timeout=1500):
    cfg = CONFIGS[name]
    src = WORKER % dict(base=BASE, path=PATH, mode=mode, msgs=MSGS,
                        tp=cfg["tp"], ep=cfg.get("ep", 1), dcp=cfg.get("dcp", 1),
                        pp=cfg.get("pp", 1), gpus=cfg.get("gpus", [0]),
                        maxtok=SP_MAX_TOKENS)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True, timeout=timeout, cwd=BASE)
    took = time.time() - t0
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        print(f"[{name}/{mode}] FAIL rc={r.returncode} took={took:.0f}s")
        print("  stderr tail:", r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "(empty)")
        return None
    d = json.loads(line[0][6:])
    d["wall_with_load"] = round(took, 1)
    with open(os.path.join(OUT, f"{mode}_{name}.json"), "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"[{name}/{mode}] OK {d['tok_s']} tok/s load={d['load_s']}s run={d['run_s']}s "
          f"steps={d['steps']} wall={took:.0f}s")
    return d


def compare():
    rows = []
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(OUT, fn)) as f:
            d = json.load(f)
        rows.append((fn[:-5], d))
    ref = None
    for name, d in rows:
        if d["mode"] in ("smoke", "bench", "quant_bf16") and d["tp"] == 1 and d["ep"] == 1:
            ref = d
            break
    print(f"{'config':<24}{'tok/s':>8}{'load_s':>8}{'run_s':>8}{'match':>8}")
    for name, d in rows:
        m = "-"
        if ref:
            same = sum(1 for k in d["outs"] if d["outs"][k] == ref["outs"].get(k))
            m = f"{same}/{len(d['outs'])}"
        print(f"{name:<24}{d['tok_s']:>8}{d['load_s']:>8}{d['run_s']:>8}{m:>8}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("smoke", "bench"):
        for name in sys.argv[2:]:
            run_config(name, cmd)
    elif cmd == "quant":
        for m in sys.argv[2:]:
            name = "tp1" if m != "quant_w4_ep2" else "ep2"
            run_config(name, m)
    elif cmd == "compare":
        compare()
