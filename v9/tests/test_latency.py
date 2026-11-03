"""并发/长尾基准：记录每个请求完成延迟、P50/P95/P99 与稳态吞吐。"""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from samplingParams import SamplingParams

MODEL = os.environ.get("QWEN3_OMNI_MODEL", "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct")
if not os.path.exists(os.path.join(MODEL, "config.json")):
    MODEL = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"
SPECS = {
    "ep4": (1, 4, 1, 1, [0,1,2,3]),
    "ep8": (1, 8, 1, 1, list(range(8))),
    "tp2": (2, 1, 1, 1, [0,1]),
    "pp2": (1, 1, 1, 2, [0,1]),
}
MSGS = ["介绍一下你自己", "1+1等于几？只回答数字", "天空为什么是蓝色的？一句话",
        "写一个 Python 快速排序", "翻译成英文：今天天气真好", "列举三种水果",
        "请解释分布式推理中张量并行、流水线并行、专家并行的通信边界、显存占用和性能权衡。" * 4,
        "请给出一个面向生产环境的 MoE 服务压测方案，包含请求分布、长尾指标、专家热点、通信带宽和故障恢复。" * 3]

def pct(vals, q):
    if not vals: return 0.0
    s = sorted(vals); i = min(len(s)-1, max(0, int(round((len(s)-1)*q))))
    return s[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=SPECS, required=True)
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--heterogeneous", action="store_true",
                    help="alternate 4/8/16/32-token budgets to expose tail latency")
    ap.add_argument("--output", default="/tmp/qwen_latency.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    template = os.path.join(MODEL, "chat_template.json")
    if os.path.exists(template):
        with open(template, encoding="utf-8") as f: tok.chat_template = json.load(f)["chat_template"]
    prompts = [tok.encode(tok.apply_chat_template([{"role":"user","content":m}], tokenize=False,
                                                    add_generation_prompt=True), add_special_tokens=False)
               for m in MSGS]
    tp, ep, dcp, pp, gpus = SPECS[a.config]
    from engine.core import EngineCore
    t0 = time.perf_counter()
    core = EngineCore(MODEL, tp_size=tp, ep_size=ep, dcp_size=dcp, pp_size=pp,
                      master_port=29941 + len(gpus), gpus=gpus,
                      config_kwargs={"max_num_seqs":64, "max_model_len":1024,
                                     "max_num_batched_tokens":512,
                                     "gpu_memory_utilization":0.85, "enforce_eager":True})
    init_s = time.perf_counter() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    rows=[]
    for n in [int(x) for x in a.concurrency.split(",")]:
        sid2i={};
        for i in range(n):
            req_sp = (SamplingParams(temperature=0.0,
                                     max_tokens=[4, 8, 16, 32][i % 4])
                      if a.heterogeneous else sp)
            seq=core.add_request(prompts[i % len(prompts)], req_sp); sid2i[seq.seq_id]=i
        start=time.perf_counter(); done={}
        while not core.is_finished():
            for sid, toks in core.step():
                done[sid]= (time.perf_counter()-start, len(toks))
        lat=[v[0] for v in done.values()]
        nt=sum(v[1] for v in done.values()); elapsed=time.perf_counter()-start
        rows.append({"concurrency":n,"elapsed_s":round(elapsed,4),"tokens":nt,
                     "tok_s":round(nt/elapsed,3) if elapsed else 0,
                     "latency_p50_s":round(pct(lat,.50),4),
                     "latency_p95_s":round(pct(lat,.95),4),
                     "latency_p99_s":round(pct(lat,.99),4),
                     "latency_max_s":round(max(lat) if lat else 0,4),
                     "completed":len(done)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    result={"model":MODEL,"config":a.config,"tp":tp,"ep":ep,"pp":pp,"gpus":len(gpus),
            "init_s":round(init_s,3),"max_tokens":a.max_tokens,
            "heterogeneous":a.heterogeneous,
            "prompt_lengths": [len(p) for p in prompts], "rows":rows}
    with open(a.output,"w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False), flush=True)
    core.shutdown()
if __name__ == "__main__": main()
