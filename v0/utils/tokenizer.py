# utils/tokenizer.py
import json
import os

from tokenizers import Tokenizer as _HfTokenizer


class Tokenizer:
    """极薄的一层封装：用 tokenizers 库加载 tokenizer.json，提供 encode/decode 和特殊 token。"""

    def __init__(self, model_dir: str):
        path = os.path.join(model_dir, "tokenizer.json")
        assert os.path.exists(path), f"没找到 tokenizer.json: {model_dir}"
        self.tk = _HfTokenizer.from_file(path)

        # eos 优先读 generation_config.json（可能是 int 或 list）
        eos = 151645
        pad = None
        gen_cfg = os.path.join(model_dir, "generation_config.json")
        if os.path.exists(gen_cfg):
            with open(gen_cfg, encoding="utf-8") as f:
                raw = json.load(f)
            if "eos_token_id" in raw:
                eos = raw["eos_token_id"]
            pad = raw.get("pad_token_id")
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.pad_token_id = pad if pad is not None else next(iter(self.eos_ids))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tk.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tk.decode(token_ids, skip_special_tokens=skip_special_tokens)
