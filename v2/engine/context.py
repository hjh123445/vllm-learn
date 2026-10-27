# v2/engine/context.py
"""一次前向的 attention 上下文（全局单例）。

模型前向签名只剩 (input_ids, positions)——其余全部和 attention 相关的信息
（槽位映射、混合批次分组、mask）都收在这个 Context 里：
ModelRunner 每次前向 set_context 一次，每层 attention get_context 读，
模型文件不再传 attention 参数。"""
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    slot_mapping: torch.Tensor | None = None   # [total_q] 每 token 的全局槽位 (block_id*block_size+偏移)
    # ---- decode 组（s=1 天然同形，合批）----
    dec_n: int = 0
    dec_idx: torch.Tensor | None = None        # [n] 组内各序列在 flat 批里的索引
    dec_tabs: torch.Tensor | None = None       # [n, nblk] padded block 表
    dec_read_len: int = 0
    dec_mask: torch.Tensor | None = None       # [n, 1, 1, L] 加性 mask
    # ---- prefill 行（s 各异，逐行）----
    pf_spans: list | None = None               # [(i0, i1, tabs, read_len), ...]
    pf_masks: list | None = None               # [1, 1, s, L] 与 pf_spans 对齐


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(is_prefill=False, slot_mapping=None, dec_n=0, dec_idx=None,
               dec_tabs=None, dec_read_len=0, dec_mask=None, pf_spans=None, pf_masks=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, slot_mapping, dec_n, dec_idx, dec_tabs,
                       dec_read_len, dec_mask, pf_spans, pf_masks)


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
