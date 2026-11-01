"""v9 DSV4 通路验证（P0-1 修复的裁决：load_thinker_config 分派后 DSV4 能否启动）。

注意（P0-15 诚实声明）：v9 的 DSV4Attention compressor 未接入 forward
（单发模式）。本脚本只验证【装载 + 前向 + 生成不崩】——语义正确性由
v6.3 的既有数据背书，v9 的 DSV4 是骨架。
用法: python tests/dsv4_smoke.py [n_gpus]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = "/data/ifly_jyhe23/dsv4_rt/DeepSeek-V4-Flash"


def main():
    n_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    from engine.core import EngineCore

    t0 = time.time()
    core = EngineCore(PATH, tp_size=1, ep_size=n_gpus, gpus=list(range(n_gpus)),
                      config_kwargs={"max_num_seqs": 4, "max_model_len": 1024,
                                     "max_num_batched_tokens": 2048,
                                     "gpu_memory_utilization": 0.85,
                                     "enforce_eager": True})
    t_load = time.time() - t0
    print(f"[dsv4-smoke] 装载 {t_load:.0f}s（EP{n_gpus}，bf16 反量化分层分放）", flush=True)

    prompt = list(range(100, 148))          # 任意 48 个 token id（不追求语义）
    from samplingParams import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    seq = core.add_request(prompt, sp)
    t1 = time.time()
    outs = {}
    while not core.is_finished():
        for sid, toks in core.step():
            outs[sid] = toks
    wall = time.time() - t1
    core.shutdown()
    got = outs.get(seq.seq_id, [])
    print("RESULT" + json.dumps({"load_s": round(t_load, 1), "gen_s": round(wall, 2),
                                 "n_tok": len(got), "tokens": got[:8],
                                 "tok_s": round(len(got) / wall, 2) if wall else 0}))


# spawn 守卫：EngineCore 的 worker 用 spawn 启动，子进程 re-import 本模块时
# 不能重复执行构造（freeze_support 报错的根因）
if __name__ == "__main__":
    main()
