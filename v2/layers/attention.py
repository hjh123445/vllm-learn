# v2/layers/attention.py
"""v2 attention（平铺版，正常做法：按形状分组）。

平铺批按形状分两组算：
  - decode 组：全部 s=1 的序列天然同形 → 一次 gather 整组 cache（read_batch），
    一条 [n, h, 1, L] 的批注意力。这是热路径（每步都跑），必须组批；
  - prefill 行：s 各不相同没法组批 → 逐行 read_seq 算 [1, h, s_i, L_i]。
    prefill 每个请求只发生一次，不在热路径上，逐行可接受。

cache 写仍是 write_flat：整个平铺批按 token 一次散射（每层 1 次 kernel）。
真·变长 kernel（cu_seqlens 单 kernel 全算）留 v3。"""
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, kv, layer_idx, num_head, num_kv_head, head_dim, scale):
        super().__init__()
        self.kv = kv                    # KVCacheManager，全模型共享一个
        self.layer_idx = layer_idx      # 本层在池子里的切片号
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.head_dim = head_dim
        self.scale = scale

    def forward(self, q, k_new, v_new, plan) -> torch.Tensor:
        """q: [total_q, num_head, head_dim]（已 rope）；k/v_new 同为平铺。
        返回 [total_q, num_head * head_dim]（o_proj 输入）。"""
        T = q.shape[0]
        n_rep = self.num_head // self.num_kv_head
        self.kv.write_flat(self.layer_idx, plan.tok_blk, plan.tok_off, k_new, v_new)
        out = q.new_empty(T, self.num_head * self.head_dim)

        # ---- decode 组：s=1 天然同形，一次 gather + 一条批注意力 ----
        d = plan.dec
        if d is not None:
            qd = q[d.idx]                                          # [n, h, hd]
            kd, vd = self.kv.read_batch(self.layer_idx, d.tabs, d.read_len)   # [n, kvh, L, hd]
            kd = kd.repeat_interleave(n_rep, dim=1)
            vd = vd.repeat_interleave(n_rep, dim=1)
            scores = (qd.unsqueeze(2) @ kd.transpose(-2, -1)) * self.scale   # [n, h, 1, L]
            scores = scores + d.mask
            od = (scores.softmax(-1) @ vd).squeeze(2)             # [n, h, hd]
            out[d.idx] = od.reshape(d.n, -1)

        # ---- prefill 行：s 各异，逐行（不在热路径）----
        for sp in plan.spans:
            s = sp.i1 - sp.i0
            qp = q[sp.i0:sp.i1].unsqueeze(0).transpose(1, 2)      # [1, h, s, hd]
            kp, vp = self.kv.read_seq(self.layer_idx, sp.tabs, sp.read_len)   # [1, kvh, L, hd]
            kp = kp.repeat_interleave(n_rep, dim=1)
            vp = vp.repeat_interleave(n_rep, dim=1)
            scores = (qp @ kp.transpose(-2, -1)) * self.scale       # [1, h, s, L]
            scores = scores + sp.mask
            o = scores.softmax(-1) @ vp                             # [1, h, s, hd]
            out[sp.i0:sp.i1] = o.transpose(1, 2).reshape(s, -1)     # [s, h*hd]
        return out
