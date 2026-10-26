# v2/layers/attention.py
"""v2 attention（平铺版）：flattened batch，朴素地逐序列启动注意力。

每层只做三件事：
  1. write_flat: 把本步新算的 k/v 按 token 一次性散射进各请求的 block（全向量化）；
  2. 逐序列循环：每个请求各自 read_seq 自己的 block 表 → [1, kvh, L, hd]，
     一条 [1, h, s_i, L_i] 的注意力（不组批、不 padding——每个序列长度天然可以不同）；
  3. 拼回 [total_q, h*hd]。
真·变长 kernel（cu_seqlens 一个 kernel 全算）留 v3；这里按用户要求保持朴素。"""
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

        # 朴素逐序列：每个序列各自读自己的 block 表、算自己的注意力
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
