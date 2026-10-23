import torch
import torch.nn as nn


class KVCache(nn.Module):
    def __init__(self, batch_size, num_kv_head, max_position, head_dim):
        super().__init__()
        # register_buffer：随 model.to(device, dtype) 一起迁移/转精度，
        # 也不会进 state_dict 干扰对账
        self.register_buffer(
            "k", torch.zeros(batch_size, num_kv_head, max_position, head_dim),
            persistent=False)
        self.register_buffer(
            "v", torch.zeros(batch_size, num_kv_head, max_position, head_dim),
            persistent=False)
        self.cur_len = 0
        self.cur_batch = 0

    def reset(self):
        self.cur_len = 0
        self.cur_batch = 0

    def append(self, k, v):
        n = k.shape[2]
        b = k.shape[0]
        assert self.cur_len + n <= self.k.shape[2], \
            f"KV cache 溢出: {self.cur_len} + {n} > {self.k.shape[2]}"
        # 预分配的 batch 维可能大于本次实际 batch，写入/读取都要按实际 batch 切，
        # 否则 q [1,h,s,L] 与 k [8,h,L,hd] 会在 matmul 里被广播悄悄扩批
        self.k[:b, :, self.cur_len:self.cur_len + n, :] = k
        self.v[:b, :, self.cur_len:self.cur_len + n, :] = v
        self.cur_len += n
        self.cur_batch = b

    def get(self):
        b = self.cur_batch
        return (self.k[:b, :, :self.cur_len, :], self.v[:b, :, :self.cur_len, :])


class Attention(nn.Module):
    def __init__(self,
                 batch_size,
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
        self.cache = KVCache(batch_size, num_kv_head, max_position, head_dim)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b, num_head, s, head_dim]  k/v: [b, num_kv_head, s, head_dim]
        assert self.num_head % self.num_kv_head == 0
        self.cache.append(k, v)
        k, v = self.cache.get()               # [b, kvh, cur_len, head_dim]
        n_rep = self.num_head // self.num_kv_head
        k = k.repeat_interleave(n_rep, dim=1)  # GQA: kv 复制到 q 的头数
        v = v.repeat_interleave(n_rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) * self.scale   # [b, h, s, cur_len]
        if attn_mask is not None:
            scores = scores + attn_mask       # 浮点加性 mask，pad 位置 -inf
        o = scores.softmax(dim=-1) @ v        # [b, h, s, head_dim]
        return o
