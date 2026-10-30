"""Embedding / LmHead 组件（v9.3 并行化）。

VocabParallel 语义（Megatron）：
  Embedding：权重行切（每 rank 持 vocab/tp 行）。输入 id 在本 rank 段内才
  查表，段外贡献 0，全部 rank 的输出 allreduce 求和——数学上等价于全量查表；
  LmHead：权重行切（=输出的 vocab 维切片）。每 rank 输出 [T, vocab/tp] 的
  部分 logits，all_gather 拼接成全量（采样需要完整 vocab 分布）。

单卡退化：tp=1 时与全量逐位等价（allreduce/all_gather 直通），同一路径。
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from layers.parallel import STATE, shard_bounds, tp_allreduce


class Embedding(nn.Module):
    """vocab 行切的查表：本 rank 持 [vocab/tp, hidden]，跨段 id 查零 + allreduce。"""

    def __init__(self, vocab_dim, hidden_dim):
        super().__init__()
        start, length = shard_bounds(vocab_dim, STATE.tp_size, STATE.tp_rank)
        self.vocab_start, self.vocab_len = start, length
        self.weight = nn.Parameter(torch.empty(length, hidden_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded, shard_id=None):
        """loaded 全量 [vocab, hidden] → 本 rank 行段。"""
        param.data.copy_(loaded[self.vocab_start:self.vocab_start + self.vocab_len])

    def forward(self, x):
        # 段内 id 重映射到本地行；段外位置查 0（allreduce 后恰好无贡献）
        local = x - self.vocab_start
        inside = (local >= 0) & (local < self.vocab_len)
        safe = local.clamp(0, self.vocab_len - 1)
        out = F.embedding(safe, self.weight) * inside.unsqueeze(-1).to(self.weight.dtype)
        return tp_allreduce(out)


class LmHead(nn.Module):
    """vocab 行切的输出投影：本 rank 持 [vocab/tp, hidden]，输出 gather 拼全量。"""

    def __init__(self, hidden_dim, vocab_dim):
        super().__init__()
        start, length = shard_bounds(vocab_dim, STATE.tp_size, STATE.tp_rank)
        self.vocab_start, self.vocab_len = start, length
        self.weight = nn.Parameter(torch.empty(length, hidden_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded, shard_id=None):
        param.data.copy_(loaded[self.vocab_start:self.vocab_start + self.vocab_len])

    def forward(self, x):
        part = F.linear(x, self.weight)          # [T, vocab/tp]
        if STATE.tp_group is not None and dist.is_available() and dist.is_initialized():
            parts = [torch.empty_like(part) for _ in range(STATE.tp_size)]
            dist.all_gather(parts, part.contiguous(), group=STATE.tp_group)
            return torch.cat(parts, dim=-1)      # [T, vocab]
        return part
