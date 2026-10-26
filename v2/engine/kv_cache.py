# v2/engine/kv_cache.py
"""v2 显存管理中枢：paged KV cache。

池子 [num_layers, num_blocks, num_kv_heads, block_size, head_dim]，16 token 一个 block，
每请求一张 block table。池子大小由 warmup 量出来（见 model_runner）。

读写接口是平铺的（flattened batch）：按 token 写（write_flat），按序列读（read_seq）。
池子大小、block 分配/回收/抢占水位全在这里，scheduler 只跟它要 block。
"""
import torch


class KVCacheManager:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 block_size: int, max_num_blocks: int, dtype, device):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.dtype, self.device = dtype, device
        self._alloc_pool(max_num_blocks)
        self.tables: dict[int, list[int]] = {}    # req_id -> block 列表（逻辑序 = token 序）

    def _alloc_pool(self, n_blocks: int):
        shape = (self.num_layers, n_blocks, self.num_kv_heads, self.block_size, self.head_dim)
        self.k = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.free_blocks = list(range(n_blocks))

    def rebuild(self, max_num_blocks: int):
        """warmup profile 后按量出来的大小重建池子。tables 必须为空。"""
        assert not self.tables, "还有存活请求，不能重建池子"
        self.max_num_blocks = max_num_blocks
        self._alloc_pool(max_num_blocks)

    # ============ 账本 ============

    @property
    def block_bytes(self) -> int:
        return self.k[:, 0].numel() * self.k.element_size() * 2

    def pool_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def nblocks(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def num_free(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, n_tokens: int, reserve_blocks: int = 0) -> bool:
        """准入判据：要得起 prompt 的 block 且留得住水位储备。"""
        return len(self.free_blocks) >= self.nblocks(n_tokens) + reserve_blocks

    # ============ 生命周期（scheduler 调用）============

    def allocate(self, req_id: int, n_tokens: int) -> list[int]:
        need = self.nblocks(n_tokens)
        assert need <= len(self.free_blocks), f"block 不足: 要 {need} 剩 {len(self.free_blocks)}"
        self.tables[req_id] = [self.free_blocks.pop() for _ in range(need)]
        return self.tables[req_id]

    def ensure_append(self, req_id: int, pos: int) -> bool:
        """decode 写位置 pos 前检查：pos 落在新 block 起点则需要新 block。耗尽返回 False。"""
        if pos % self.block_size == 0:
            if not self.free_blocks:
                return False
            self.tables[req_id].append(self.free_blocks.pop())
        return True

    def free(self, req_id: int):
        self.free_blocks.extend(self.tables.pop(req_id, []))

    # ============ block 表视图 ============

    def block_table(self, req_id: int, n_blocks: int) -> torch.Tensor:
        """单个请求的 block 表切片 [n_blocks]（GPU）。表不足补 0（反正会被 mask 挡住）。"""
        tab = self.tables.get(req_id, [])
        t = torch.zeros(n_blocks, dtype=torch.long)
        t[:min(len(tab), n_blocks)] = torch.tensor(tab[:n_blocks], dtype=torch.long)
        return t.to(self.k.device)

    # ============ 物理读写（attention 每层调，全向量化零循环零同步）============

    def write_flat(self, layer_idx: int, tok_blk: torch.Tensor, tok_off: torch.Tensor,
                   k_new: torch.Tensor, v_new: torch.Tensor):
        """按 token 散射写整个平铺批。tok_blk/tok_off: [total_q] 每 token 的目标
        block 号和块内偏移（plan 预先查好表）。k_new: [total_q, kvh, hd]。"""
        T, H, D = k_new.shape
        bs = self.block_size
        b = tok_blk.long().view(T, 1, 1)
        o = tok_off.long().view(T, 1, 1)
        h = torch.arange(H, device=k_new.device).view(1, H, 1)
        d = torch.arange(D, device=k_new.device).view(1, 1, D)
        lin = (b * (H * bs * D) + h * (bs * D) + o * D + d).reshape(-1)
        self.k[layer_idx].view(-1)[lin] = k_new.reshape(-1)
        self.v[layer_idx].view(-1)[lin] = v_new.reshape(-1)

    def read_seq(self, layer_idx: int, tabs: torch.Tensor, read_len: int):
        """按序列读一个请求的 cache → [1, kvh, read_len, hd]（prefill 行用，行数少）。
        只 gather 需要的 block，尾 block 内多出的槽位由调用方的 mask 挡住。"""
        lay_k = self.k[layer_idx][tabs]               # [nblk, kvh, bs, hd]
        lay_v = self.v[layer_idx][tabs]
        nblk, H, bs, D = lay_k.shape
        k = lay_k.permute(1, 0, 2, 3).reshape(H, nblk * bs, D)[:, :read_len]
        v = lay_v.permute(1, 0, 2, 3).reshape(H, nblk * bs, D)[:, :read_len]
        return k.unsqueeze(0), v.unsqueeze(0)

    def read_batch(self, layer_idx: int, tabs: torch.Tensor, read_len: int):
        """整组一次 gather：tabs [n, nblk]（padded 补 0）→ [n, kvh, read_len, hd]。
        decode 组用：n 个序列一次 gather 拷贝，替代 n 次 read_seq。
        多出的槽位由调用方的 mask 挡住。"""
        lay_k = self.k[layer_idx][tabs]               # [n, nblk, kvh, bs, hd]
        lay_v = self.v[layer_idx][tabs]
        n, nblk, H, bs, D = lay_k.shape
        k = lay_k.permute(0, 2, 1, 3, 4).reshape(n, H, nblk * bs, D)[:, :, :read_len]
        v = lay_v.permute(0, 2, 1, 3, 4).reshape(n, H, nblk * bs, D)[:, :, :read_len]
        return k, v
