# v1/layers/attention.py
"""v1 混合批次版 attention（Orca 式：prefill 和 decode 在同一个 forward 里跑）。

外部形状是 flattened batch：没有 [b, s] 两个维度，所有 query token 摊平成
[total_q, hidden]——非注意力层（norm/MLP/投影）对 total_q 无差别，天然兼容。

attention 内部把 flat 批按 plan 分两组算：
  - decode 组：全部是 s=1 的请求 → 天然同形，按行合批成 [n, h, 1, L]
  - prefill 行：每行 s 各不相同 → 逐行算 [1, h, s_i, L_i]（cache 是 view 零拷贝）

真·变长 kernel（flash-attn varlen / FlashInfer：flattened + cu_seqlens 一个 kernel
算完全部行）是 v3 的活；这里先把调度语义做对。
"""
import torch
import torch.nn as nn


class KVCache(nn.Module):
    """[max_requests, num_kv_head, max_position, head_dim]，每行一个请求。
    v1 的 cache 槽位 = 请求自己的真实位置（prompt 从 0 写，decode 接着写）。"""

    def __init__(self, max_requests, num_kv_head, max_position, head_dim):
        super().__init__()
        self.max_position = max_position
        self.register_buffer(
            "k", torch.zeros(max_requests, num_kv_head, max_position, head_dim),
            persistent=False)
        self.register_buffer(
            "v", torch.zeros(max_requests, num_kv_head, max_position, head_dim),
            persistent=False)

    def write_flat(self, rows: torch.Tensor, positions: torch.Tensor,
                   k_new: torch.Tensor, v_new: torch.Tensor):
        """按 token 散射写（不是按行）：rows/positions 是 [total_q]，k_new [total_q, kvh, hd]。
        线性下标一次 scatter，零 python 循环零同步。"""
        T, H, D = k_new.shape
        P = self.k.shape[2]
        r = rows.long().view(T, 1, 1)
        p = positions.long().view(T, 1, 1)
        h = torch.arange(H, device=k_new.device).view(1, H, 1)
        d = torch.arange(D, device=k_new.device).view(1, 1, D)
        lin = (r * (H * P * D) + h * (P * D) + p * D + d).reshape(-1)
        self.k.view(-1)[lin] = k_new.reshape(-1)
        self.v.view(-1)[lin] = v_new.reshape(-1)

    def read_rows(self, rows, length: int):
        """rows: [n] tensor → gather 拷贝 [n, kvh, length, hd]；rows: int → view [1, kvh, length, hd]。"""
        if torch.is_tensor(rows):
            return self.k[rows, :, :length], self.v[rows, :, :length]
        return self.k[rows:rows + 1, :, :length], self.v[rows:rows + 1, :, :length]


class Attention(nn.Module):
    def __init__(self, max_requests, max_position, num_head, num_kv_head, head_dim, scale):
        super().__init__()
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.head_dim = head_dim
        self.scale = scale
        self.cache = KVCache(max_requests, num_kv_head, max_position, head_dim)

    def forward(self, q, k_new, v_new, plan) -> torch.Tensor:
        """q: [total_q, num_head, head_dim]（已 rope）；k/v_new: [total_q, num_kv_head, head_dim]。
        返回 [total_q, num_head * head_dim]（o_proj 的输入）。"""
        T = q.shape[0]
        n_rep = self.num_head // self.num_kv_head
        self.cache.write_flat(plan.row_ids, plan.positions, k_new, v_new)
        out = q.new_empty(T, self.num_head * self.head_dim)

        # ---- decode 组：s=1 天然同形，按行合批 ----
        if plan.n_dec > 0:
            qd = q[plan.dec_idx]                                        # [n, h, hd]
            kd, vd = self.cache.read_rows(plan.dec_rows, plan.dec_read_len)
            kd = kd.repeat_interleave(n_rep, dim=1)
            vd = vd.repeat_interleave(n_rep, dim=1)
            scores = (qd.unsqueeze(2) @ kd.transpose(-2, -1)) * self.scale   # [n, h, 1, L]
            scores = scores + plan.dec_mask
            od = (scores.softmax(-1) @ vd).squeeze(2)                   # [n, h, hd]
            out[plan.dec_idx] = od.reshape(plan.n_dec, -1)

        # ---- prefill 行：s 各异，逐行（行内 cache 是 view，零拷贝）----
        for (i0, i1, row, L), mask in zip(plan.pf_spans, plan.pf_masks):
            qp = q[i0:i1].unsqueeze(0).transpose(1, 2)               # [1, h, s, hd]
            kp, vp = self.cache.read_rows(row, L)                     # [1, kvh, L, hd]
            kp = kp.repeat_interleave(n_rep, dim=1)
            vp = vp.repeat_interleave(n_rep, dim=1)
            scores = (qp @ kp.transpose(-2, -1)) * self.scale         # [1, h, s, L]
            scores = scores + mask
            op = scores.softmax(-1) @ vp                              # [1, h, s, hd]
            out[i0:i1] = op.transpose(1, 2).reshape(i1 - i0, -1)       # [s, h*hd]

        return out
