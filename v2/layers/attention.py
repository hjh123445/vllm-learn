# v2/layers/attention.py
"""v2 attention（平铺版，读全局 Context）。

每层只做三件事：
  1. write_slot: 把本步新算的 k/v 按全局槽位一次散射进各请求的 block；
  2. decode 组合批（s=1 天然同形，热路径）+ prefill 逐行；
  3. 拼回 [total_q, h*hd]。
attention 参数里不再有 plan——全部从 get_context() 读。"""
import torch
import torch.nn as nn

from engine.context import get_context


class Attention(nn.Module):
    def __init__(self, kv, layer_idx, num_head, num_kv_head, head_dim, scale):
        super().__init__()
        self.kv = kv                    # KVCacheManager，全模型共享一个
        self.layer_idx = layer_idx      # 本层在池子里的切片号
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.head_dim = head_dim
        self.scale = scale

    def forward(self, q, k_new, v_new) -> torch.Tensor:
        """q/k/v: [total_q, heads, head_dim]（已 rope，平铺无 b/s 维）。
        返回 [total_q, num_head * head_dim]（o_proj 输入）。"""
        ctx = get_context()
        T = q.shape[0]
        n_rep = self.num_head // self.num_kv_head
        self.kv.write_slot(self.layer_idx, ctx.slot_mapping, k_new, v_new)
        out = q.new_empty(T, self.num_head * self.head_dim)

        # ---- decode 组：s=1 天然同形，一次 gather + 一条批注意力 ----
        if ctx.dec_n > 0:
            qd = q[ctx.dec_idx]                                          # [n, h, hd]
            kd, vd = self.kv.read_batch(self.layer_idx, ctx.dec_tabs, ctx.dec_read_len)
            kd = kd.repeat_interleave(n_rep, dim=1)
            vd = vd.repeat_interleave(n_rep, dim=1)
            scores = (qd.unsqueeze(2) @ kd.transpose(-2, -1)) * self.scale   # [n, h, 1, L]
            scores = scores + ctx.dec_mask
            od = (scores.softmax(-1) @ vd).squeeze(2)                     # [n, h, hd]
            out[ctx.dec_idx] = od.reshape(ctx.dec_n, -1)

        # ---- prefill 行：s 各异，逐行（不在热路径）----
        for (i0, i1, tabs, read_len), mask in zip(ctx.pf_spans, ctx.pf_masks):
            s = i1 - i0
            qp = q[i0:i1].unsqueeze(0).transpose(1, 2)      # [1, h, s, hd]
            kp, vp = self.kv.read_seq(self.layer_idx, tabs, read_len)     # [1, kvh, L, hd]
            kp = kp.repeat_interleave(n_rep, dim=1)
            vp = vp.repeat_interleave(n_rep, dim=1)
            scores = (qp @ kp.transpose(-2, -1)) * self.scale             # [1, h, s, L]
            scores = scores + mask
            o = scores.softmax(-1) @ vp                                   # [1, h, s, hd]
            out[i0:i1] = o.transpose(1, 2).reshape(s, -1)                 # [s, h*hd]
        return out
