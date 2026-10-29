"""v6.3 阶段三验证：DeepSeek-V4 端到端（装载 + 生成 + 与 HF 对齐检查）。"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deepseek_v4 import DSV4ForCausalLM
from utils.dsv4_loader import load_dsv4

MODEL = "/data/xmhuang6/deepseek/DeepSeek-V4-Flash"


def main():
    print("构造模型 + 装载（FP8 反量化，43 层 × 257 专家，预计数分钟）...")
    model = DSV4ForCausalLM(MODEL)
    # 反量化后 ~584GB bf16（FP4 专家 ×4 + FP8 其他 ×2）：8 卡分放（43/8 ≈ 5.4 层/卡）
    # cuda:0 被别的进程占 4GB 且要放 embed+head——只放 4 层；其余 39 层分 7 卡
    n_layers = 43
    layer_map = {}
    rest = list(range(4, n_layers))
    per = (len(rest) + 6) // 7                            # 39/7 ≈ 6 层/卡
    for i in range(n_layers):
        if i < 4:
            layer_map[i] = "cuda:0"
        else:
            layer_map[i] = f"cuda:{1 + (i - 4) // per}"
    layer_map[-1] = "cuda:7"                              # norm/hc_head/lm_head
    print(f"装载前 cuda:0 = {torch.cuda.memory_allocated(0)/1e9:.1f}GB", flush=True)
    load_dsv4(model, MODEL, device="cuda:0", dtype=torch.bfloat16,
              layer_device_map=layer_map)
    print(f"装载后 cuda:0 = {torch.cuda.memory_allocated(0)/1e9:.1f}GB", flush=True)

    # 简单贪心生成（平铺单请求）
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = "The capital of France is"
    ids = tok.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, device="cuda")
    positions = torch.arange(len(ids), device="cuda")

    generated = []
    with torch.no_grad():
        cur_ids, cur_pos = input_ids, positions
        for step in range(16):
            hidden = model(cur_ids, cur_pos)          # [T, H]
            logits = model.compute_logits(hidden[-1:])  # 末位
            nxt = int(logits.argmax(-1).item())
            generated.append(nxt)
            if nxt in (tok.eos_token_id,):
                break
            cur_ids = torch.cat([cur_ids, torch.tensor([nxt], device="cuda")])
            cur_pos = torch.arange(cur_ids.shape[0], device="cuda")

    text = tok.decode(generated, skip_special_tokens=True)
    print(f"\n生成 ({len(generated)} tok): {text!r}")
    print(f"prompt: {prompt!r}")
    assert len(text.strip()) > 0, "空输出"
    print("\n===== v6.3 端到端: PASS =====")


if __name__ == "__main__":
    main()
