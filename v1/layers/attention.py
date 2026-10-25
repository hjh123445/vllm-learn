# v1/layers/attention.py
"""v1 迭代级调度版本的 attention：
KV cache 每行一个请求（行号 = scheduler 分配的 slot），
每步的 batch 是当前活跃请求的子集，靠 cache_rows 索引各自的 cache 行。
非注意力层不需要任何改动——它们只看 [b, s, hidden]，batch 行是谁无所谓。"""
import torch
import torch.nn as nn


class KVCache(nn.Module):
    """[max_requests, num_kv_head, max_position, head_dim]，每行一个请求。
    请求结束后 slot 回收复用：新请求只读自己写过的前缀（mask 保证），脏数据不会被读到。"""

    def __init__(self, max_requests, num_kv_head, max_position, head_dim):
        super().__init__()
        self.max_position = max_position
        self.register_buffer(
            "k", torch.zeros(max_requests, num_kv_head, max_position, head_dim),
            persistent=False)
        self.register_buffer(
            "v", torch.zeros(max_requests, num_kv_head, max_position, head_dim),
            persistent=False)

    def write(self, rows, starts, k_new, v_new):
        """rows/starts: [b]；k_new: [b, kvh, s, hd]，从各行 start 起连续写入 s 个位置"""
        s = k_new.shape[2]
        for i in range(k_new.shape[0]):
            pos = int(starts[i])
            row = int(rows[i])
            self.k[row, :, pos:pos + s, :] = k_new[i]
            self.v[row, :, pos:pos + s, :] = v_new[i]

    def read(self, rows, length):
        """rows: [b] → [b, kvh, length, hd]。
        高级索引产生拷贝（性能 TODO：v2 换 paged attention 消掉这个 gather）"""
        return self.k[rows, :, :length, :], self.v[rows, :, :length, :]


class Attention(nn.Module):
    def __init__(self,
                 max_requests,
                 max_position,
                 num_head,
                 num_kv_head,
                 head_dim,
                 scale):
        super().__init__()
        self.num_head = num_head
        self.num_kv_head = num_kv_head
        self.head_dim = head_dim
        self.scale = scale
        self.cache = KVCache(max_requests, num_kv_head, max_position, head_dim)

    def forward(self, q, k_new, v_new, cache_rows, write_pos, attn_mask):
        """
        q      : [b, num_head, s, head_dim]
        k/v_new: [b, num_kv_head, s, head_dim] 当前步新算出的（待写入 cache）
        cache_rows: [b] 各行对应的 cache 行号（请求 slot）
        write_pos  : [b] 各行新 token 的写入起点
        attn_mask  : [b, 1, s, L] 加性 mask，L = 读取长度（读多少由调用方决定）
        """
        assert self.num_head % self.num_kv_head == 0
        self.cache.write(cache_rows, write_pos, k_new, v_new)
        read_len = attn_mask.shape[-1]
        k, v = self.cache.read(cache_rows, read_len)         # [b, kvh, L, hd]
        n_rep = self.num_head // self.num_kv_head
        k = k.repeat_interleave(n_rep, dim=1)                # GQA 复制
        v = v.repeat_interleave(n_rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) * self.scale      # [b, h, s, L]
        scores = scores + attn_mask
        return scores.softmax(dim=-1) @ v
