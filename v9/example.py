"""Text-only offline generation; parallelism uses the same EngineCore as serving."""
import argparse
import json
from pathlib import Path

from engine.core import EngineCore
from samplingParams import SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--prompt", default="介绍一下你自己")
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    core = EngineCore(args.model, tp_size=args.tp, pp_size=args.pp,
                      enable_expert_parallel=args.enable_expert_parallel,
                      config_kwargs={"max_model_len": 2048, "max_num_seqs": 8})
    try:
        tokenizer = core.tokenizer
        template = Path(args.model) / "chat_template.json"
        if template.exists():
            tokenizer.chat_template = json.loads(template.read_text(encoding="utf8"))["chat_template"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True,
            add_generation_prompt=True)
        outputs = core.generate([prompt], SamplingParams(temperature=0., max_tokens=args.max_tokens))
        for _, tokens in outputs:
            print(tokenizer.decode(tokens, skip_special_tokens=True))
    finally:
        core.shutdown()


if __name__ == "__main__":
    main()
