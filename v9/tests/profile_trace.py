"""为 Qwen3-Omni-30B thinker 生成可在 Chrome trace/TensorBoard 查看器打开的 CUDA trace。

单卡 EngineCore 直接在当前进程执行，因而 profiler 能看到 CUDA kernel；多进程
TP/PP 的 CUDA trace 需要分别在 worker 内启动 profiler，不能由父进程跨进程捕获。
事件名包含 stage/layer/attention/moe/router/dispatch/expert/combine、微批和 lane。
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from samplingParams import SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "QWEN3_OMNI_MODEL", "/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct"))
    ap.add_argument("--output", default="/tmp/qwen3_omni_trace.json")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(args.model, "config.json")):
        args.model = "/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct"

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "用一句话介绍你自己。"}],
        tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    from engine.core import EngineCore
    core = EngineCore(args.model, gpus=[0], master_port=29931,
                      config_kwargs={"max_num_seqs": 1, "max_model_len": 512,
                                     "max_num_batched_tokens": 512,
                                     "gpu_memory_utilization": 0.80,
                                     "enforce_eager": True})
    sp = SamplingParams(temperature=0.0, max_tokens=4)
    core.add_request(ids, sp)
    # Prefill/decode warmup is outside the captured window.
    core.step()
    trace_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(trace_dir, exist_ok=True)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True, with_stack=False) as prof:
        while not core.is_finished():
            with torch.autograd.profiler.record_function(
                    "config=Qwen3-Omni-30B;pp_stage=0;microbatch=0;lane=0"):
                core.step()
            prof.step()
    core.shutdown()
    prof.export_chrome_trace(args.output)
    meta = {
        "model": args.model,
        "parallel": {"tp": 1, "ep": 1, "pp": 1, "dp": 1},
        "events": "stageN.layerM.attention/moe; moe.router/dispatch/expert/combine",
        "cuda_activities": True,
        "generated_at": time.time(),
        "trace": args.output,
    }
    with open(args.output + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
