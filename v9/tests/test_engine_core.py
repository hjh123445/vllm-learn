"""v9.4a e2e：EngineCore + WorkerProc 架构验证（在 172.28.30.56 上跑）。

三相位裁决：
  1. UniProc（单卡，ModelRunner 同进程直调）——基线
  2. Multiproc TP2（spawn 2 个 WorkerProc，MQ 广播）
  3. 对比：greedy token 逐位一致

进程拓扑验证点：worker 进程数 == world_size；scheduler 只在 EngineCore 进程
（worker 的 forward 输入完全来自 MQ 消息）。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH = os.environ.get("MODEL_PATH",
                      "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct")
if not os.path.exists(f"{PATH}/config.json"):
    PATH = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"

from transformers import AutoTokenizer
from samplingParams import SamplingParams


def run_phase(tag, **kw):
    from engine.core import EngineCore
    tok = AutoTokenizer.from_pretrained(PATH)
    try:
        with open(f"{PATH}/chat_template.json") as f:
            tok.chat_template = json.load(f)["chat_template"]
    except FileNotFoundError:
        pass
    msgs = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话"]
    ps = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False,
                                  add_generation_prompt=True) for m in msgs]
    prompts = [tok.encode(p, add_special_tokens=False) for p in ps]

    t0 = time.time()
    kw.setdefault("gpus", [0])
    kw.setdefault("config_kwargs", {})
    kw["config_kwargs"].update({"max_num_seqs": 8, "max_model_len": 2048,
                                "gpu_memory_utilization": 0.85, "enforce_eager": True})
    core = EngineCore(PATH, **kw)
    t_init = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    outs = {}
    sid2idx = {}
    for i, p in enumerate(prompts):
        seq = core.add_request(p, sp)
        sid2idx[seq.seq_id] = i
    t0 = time.time()
    while not core.is_finished():
        for seq_id, toks in core.step():
            outs[sid2idx[seq_id]] = toks         # 按请求下标对齐（seq_id 跨 phase 累计）
    t_run = time.time() - t0
    n_tok = sum(len(t) for t in outs.values())
    print(f"[{tag}] init={t_init:.0f}s run={t_run:.1f}s tok={n_tok} "
          f"({n_tok/t_run:.1f} tok/s) 进程数={core.world_size}", flush=True)
    core.shutdown()
    # 按请求顺序返回（scheduler 完成序与提交序可能不同——用 seq_id 对齐）
    return prompts, outs, n_tok


def main():
    prompts, base_out, _ = run_phase("UniProc-单卡", tp_size=1, gpus=[0])
    import torch, gc
    # phase1 显存随 core.shutdown + gc 释放
    gc.collect(); torch.cuda.empty_cache()          # 释放 phase1 的显存

    _, tp_out, _ = run_phase("Multiproc-TP2", tp_size=2, gpus=[1, 2])

    print("\n===== v9.4a 裁决（EngineCore/WorkerProc vs 单卡直调）=====")
    base_ids = sorted(base_out.keys())
    same = all(base_out[k] == tp_out.get(k) for k in base_ids)
    n_eq = sum(1 for k in base_ids if base_out[k] == tp_out.get(k))
    print(f"  UniProc vs TP2(workers): {n_eq}/{len(base_ids)} 请求完全一致 "
          f"({'ALL MATCH' if same else 'MISMATCH'})")
    if not same:
        for k in base_ids:
            if base_out[k] != tp_out.get(k):
                j = next(i for i in range(min(len(base_out[k]), len(tp_out.get(k, []))))
                         if base_out[k][i] != tp_out.get(k, [])[i])
                print(f"    seq{k} 首个分歧 @tok{j}: {base_out[k][j]} vs {tp_out[k][j]}")
    print("\n===== v9.4a 完成 =====")


if __name__ == "__main__":
    main()
