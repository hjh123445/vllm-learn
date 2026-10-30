"""v9.4 八卡并行矩阵实验（在 172.28.30.56 上跑，9:00 前停止）。

每个 config 用【独立子进程】跑（EngineCore + workers 整体退出 → 显存干净——
同进程串行会因上一 config 的 allocator 残留 OOM）。

配置: base / tp2 / tp4 / tp8 / ep2 / ep4 / ep8 / tp2ep2 / tp4ep2 / tp2ep4 /
pp2 / pp2tp2 / pp4 / dcp2 / dcp4 / tp2dcp2ep2
用法: python tests/test_matrix.py [configs...]   # 逐个跑，结果增量存 JSON
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
if not os.path.exists(f"{PATH}/config.json"):
    PATH = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"

from transformers import AutoTokenizer
from samplingParams import SamplingParams

MSGS = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话",
        "写一个 Python 快速排序", "翻译成英文：今天天气真好", "列举三种水果"]
NEW = 48

SPECS = {
    "base":       ("TP1-基线", 1, 1, 1, 1, [0]),
    "tp2":        ("TP2", 2, 1, 1, 1, [0, 1]),
    "tp4":        ("TP4", 4, 1, 1, 1, [0, 1, 2, 3]),
    "tp8":        ("TP8", 8, 1, 1, 1, list(range(8))),
    "ep2":        ("EP2", 1, 2, 1, 1, [0, 1]),
    "ep4":        ("EP4", 1, 4, 1, 1, [0, 1, 2, 3]),
    "ep8":        ("EP8", 1, 8, 1, 1, list(range(8))),
    "tp2ep2":     ("TP2×EP2", 2, 2, 1, 1, [0, 1, 2, 3]),
    "tp4ep2":     ("TP4×EP2", 4, 2, 1, 1, list(range(8))),
    "tp2ep4":     ("TP2×EP4", 2, 4, 1, 1, list(range(8))),
    "pp2":        ("PP2", 1, 1, 1, 2, [0, 1]),
    "pp2tp2":     ("PP2×TP2", 2, 1, 1, 2, [0, 1, 2, 3]),
    "pp4":        ("PP4", 1, 1, 1, 4, [0, 1, 2, 3]),
    "dcp2":       ("TP2×DCP2", 2, 1, 2, 1, [0, 1, 2, 3]),
    "dcp4":       ("TP2×DCP4", 2, 1, 4, 1, list(range(8))),
    "tp2dcp2ep2": ("TP2×DCP2×EP2", 2, 2, 2, 1, list(range(8))),
}


def get_prompts():
    tok = AutoTokenizer.from_pretrained(PATH)
    try:
        with open(f"{PATH}/chat_template.json") as f:
            tok.chat_template = json.load(f)["chat_template"]
    except FileNotFoundError:
        pass
    ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                                  add_generation_prompt=True) for m in MSGS]
    return [tok.encode(p, add_special_tokens=False) for p in ps]


def run_one(cfg):
    tag, tp, ep, dcp, pp, gpus = SPECS[cfg]
    from engine.core import EngineCore
    prompts = get_prompts()
    t0 = time.time()
    core = EngineCore(PATH, tp_size=tp, ep_size=ep, dcp_size=dcp, pp_size=pp,
                      master_port=29711, gpus=gpus,
                      config_kwargs={"max_num_seqs": 8, "max_model_len": 2048,
                                     "gpu_memory_utilization": 0.85,
                                     "enforce_eager": True})
    t_init = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=NEW)
    sid2idx = {}
    for i, p in enumerate(prompts):
        seq = core.add_request(p, sp)
        sid2idx[seq.seq_id] = i
    outs = {}
    t0 = time.time()
    while not core.is_finished():
        for seq_id, toks in core.step():
            outs[sid2idx[seq_id]] = toks
    t_run = time.time() - t0
    n_tok = sum(len(t) for t in outs.values())
    core.shutdown()
    return {"tag": tag, "tp": tp, "ep": ep, "dcp": dcp, "pp": pp,
            "n_gpu": len(gpus), "init_s": round(t_init, 1),
            "run_s": round(t_run, 1), "tok_s": round(n_tok / t_run, 1),
            "n_tok": n_tok, "outs": {str(k): v for k, v in outs.items()}}


def main():
    which = sys.argv[1:]
    results = {}
    if os.path.exists("/tmp/v94_matrix.json"):
        try:
            results = json.load(open("/tmp/v94_matrix.json"))
        except ValueError:
            results = {}
    for cfg in which:
        try:
            import subprocess
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--one", cfg],
                capture_output=True, text=True, timeout=900)
            if p.returncode != 0:
                err = p.stderr.strip().splitlines()
                print(f"[{cfg}] FAILED rc={p.returncode}: "
                      f"{err[-1][:150] if err else '?'}", flush=True)
                continue
            js = [l for l in p.stdout.splitlines() if l.startswith("{")]
            r = json.loads(js[-1])
            print(f"[{r['tag']}] init={r['init_s']}s run={r['run_s']}s "
                  f"tok={r['n_tok']} ({r['tok_s']} tok/s) workers={r['n_gpu']}",
                  flush=True)
            results[cfg] = r
        except Exception as e:
            print(f"[{cfg}] FAILED: {e}", flush=True)
        with open("/tmp/v94_matrix.json", "w") as f:
            json.dump(results, f, ensure_ascii=False)

    if "base" in results:
        base = results["base"]["outs"]
        print("\n===== 矩阵裁决（vs 单卡基线, 6 请求 greedy）=====")
        for cfg in which:
            r = results.get(cfg)
            if not r or cfg == "base":
                continue
            outs = r["outs"]
            n_eq = sum(1 for k in base if outs.get(k) == base[k])
            print(f"  {r['tag']:<16} {n_eq}/{len(base)} 一致 "
                  f"({r['tok_s']} tok/s, {r['n_gpu']} GPU)")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        print(json.dumps(run_one(sys.argv[2])))
    else:
        main()
