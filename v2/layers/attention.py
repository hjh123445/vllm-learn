# v2/layers/attention.py
"""v2 attention：自己不持有任何显存。cache 的物理布局全在 KVCacheManager，
每层只拿 (manager, layer_idx) 视图做 paged 读写 + 注意力计算。
非注意力层与 v1 完全一致（它们不碰 cache）。"""
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, kv, layer_idx, num_head, num_kv_head, head_dim, scale):
        super().__init__()
        self.kv = kv                    # KVCacheManager，全模型共享一个
        self.layer_idx = layer_idx     # 本层在池子里的切片号
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.head_dim = head_dim
        self.scale = scale

    def forward(self, q, k_new, v_new, tabs, positions, attn_mask):
        """
        q      : [b, num_head, s, head_dim]
        k/v_new: [b, num_kv_head, s, head_dim] 当前步新算出的
        tabs   : [b, nblk] 各行的 block 表切片；positions: [b, s] 各 token 全局位置
        attn_mask: [b, 1, s, L] 加性 mask，L = read_len
        """
        assert self.num_head % self.num_kv_head == 0
        read_len = attn_mask.shape[-1]
        self.kv.write(self.layer_idx, tabs, positions, k_new, v_new)   # 先写后读
        k, v = self.kv.read(self.layer_idx, tabs, read_len)            # [b, kvh, L, hd]
        n_rep = self.num_head // self.num_kv_head
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) * self.scale               # [b, h, s, L]
        scores = scores + attn_mask
        return scores.softmax(dim=-1) @ v
