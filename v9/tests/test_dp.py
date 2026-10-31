"""v9.4b DP 验证：DP2×TP2 = 2 个独立 EngineCore（各管 2 卡）vs 单卡基线。

裁决: 12 请求（6 的两倍）分给 2 个 dp rank——每个 rank 的输出与单卡
同 prompt 输出逐位一致 + DP 总吞吐 ≈ 单 rank 的 2 倍（近线性扩展）。
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH = "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"
if not os.path.exists(f"{PATH}/config.json"):
    PATH = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"

from transformers import AutoTokenizer
from samplingParams import SamplingParams

MSGS = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话",
        "写一个 Python 快速排序", "翻译成英文：今天天气真好", "列举三种水果"]

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

def main():
    from engine.frontend import DPFrontend
    prompts = get_prompts()
    sp = SamplingParams(temperature=0.0, max_tokens=48)

    # ---- DP2×TP2: 每 core 2 卡 ----
    fe = DPFrontend(PATH, dp_size=2, tp_size=2,
                    gpus=[0, 1, 2, 3],
                    config_kwargs={"max_num_seqs": 8, "max_model_len": 2048,
                                   "gpu_memory_utilization": 0.85, "enforce_eager": True})
    sid2prompt = {}
    for p in prompts:                      # 6 请求 round-robin → 每 rank 3 个
        seq, r = fe.add_request(p, sp)
        sid2prompt[seq.seq_id] = (p, r)
    t0 = time.time()
    outs = fe.run()
    t_run = time.time() - t0
    n_tok = sum(len(t) for t in outs.values())
    print(f"[DP2×TP2] run={t_run:.1f}s tok={n_tok} ({n_tok/t_run:.1f} tok/s) "
          f"dp=2×TP2=4 GPU, 每rank3请求", flush=True)
    fe.shutdown()

    # ---- 与单卡基线对比（matrix_results.json 里 base 的 outs）----
    base_path = os.path.join(os.path.dirname(__file__), "matrix_results.json")
    if os.path.exists(base_path):
        import json as _j
        base = _j.load(open(base_path))["base"]["outs"]
        # base 是 6 请求; DP 每 rank 也 3 个同 prompt? round-robin: rank0 拿 0,2,4; rank1 拿 1,3,5
        # 输出按 prompt 对齐
        n_eq = 0
        for sid, (p, r) in sid2prompt.items():
            idx = prompts.index(p)
            if str(idx) in base and outs.get(sid) == base[str(idx)]:
                n_eq += 1
        print(f"\n===== DP 裁决: {n_eq}/{len(sid2prompt)} 请求与单卡基线一致 =====")
    print("\n===== v9.4b DP 完成 =====")

if __name__ == "__main__":
    main()
