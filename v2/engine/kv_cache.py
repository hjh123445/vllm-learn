# v2/engine/kv_cache.py
"""v2 显存管理中枢：paged KV cache。

池子形状 [num_layers, num_blocks, num_kv_heads, block_size, head_dim]，
16 token 一个 block，每请求一张 block table（逻辑连续、物理离散）。
池子大小不是拍脑袋的 batch_size × max_seq_len，而是 warmup 量出激活峰值后算出来的。

所有显存的分配/回收/水位都在这里——scheduler 只跟它要 block，不看物理布局。
attention 每层通过 layer_idx 视图读写，不持有任何显存。
"""
import torch


class KVCacheManager:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 block_size: int, max_num_blocks: int, dtype, device):
        self.num_layers, self.num_kv_heads, self.head_dim = num_layers, num_kv_heads, head_dim
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.dtype, self.device = dtype, device
        self._alloc_pool(max_num_blocks)
        self.tables: dict[int, list[int]] = {}     # req_id -> block 列表（逻辑顺序 = token 顺序）

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
        """一个 block = 同一 block 号在全部层 × k/v × kv头 × 16 × head_dim"""
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

    # ============ 生命周期 ============

    def allocate(self, req_id: int, n_tokens: int) -> list[int]:
        need = self.nblocks(n_tokens)
        assert need <= len(self.free_blocks), f"block 不足: 要 {need} 剩 {len(self.free_blocks)}"
        self.tables[req_id] = [self.free_blocks.pop() for _ in range(need)]
        return self.tables[req_id]

    def ensure_append(self, req_id: int, pos: int) -> bool:
        """decode 写位置 pos 前检查：pos 落在新 block 起点（pos % block_size == 0）
        则需要新 block。池子耗尽返回 False——由 scheduler 决定抢占谁。"""
        if pos % self.block_size == 0:
            if not self.free_blocks:
                return False
            self.tables[req_id].append(self.free_blocks.pop())
        return True

    def free(self, req_id: int):
        self.free_blocks.extend(self.tables.pop(req_id, []))

    # ============ batch 视图 ============

    def tables_batch(self, req_ids: list[int], n_blocks: int) -> torch.Tensor:
        """[b, n_blocks] 的 block 表切片（GPU）。表不足 n_blocks 的行补 0，
        对应位置反正会被 attention mask 挡住。"""
        t = torch.zeros(len(req_ids), n_blocks, dtype=torch.long)
        for i, r in enumerate(req_ids):
            tab = self.tables[r]
            t[i, :min(len(tab), n_blocks)] = torch.tensor(tab[:n_blocks], dtype=torch.long)
        return t.to(self.k.device)

    # ============ 物理读写（attention 每层调一次，全部向量化，零 python 循环零同步）============

    def write(self, layer_idx: int, tabs: torch.Tensor, positions: torch.Tensor,
              k_new: torch.Tensor, v_new: torch.Tensor):
        """把当前步新算出的 k/v 散射进各请求的 block。
        tabs: [b, nblk] block 表切片；positions: [b, s] 各 token 的全局位置；
        k_new: [b, kvh, s, hd]。线性下标一次 scatter，无逐行循环。"""
        b, H, s, D = k_new.shape
        bs = self.block_size
        bi = torch.arange(b, device=self.k.device)[:, None]      # [b, 1]
        blk = tabs[bi, positions // bs]                          # [b, s] 每个 token 落在哪个 block
        off = positions % bs                                     # [b, s] block 内偏移
        lay_k, lay_v = self.k[layer_idx], self.v[layer_idx]   # [blocks, kvh, bs, hd] 视图
        h_idx = torch.arange(H, device=self.k.device).view(1, H, 1, 1)
        d_idx = torch.arange(D, device=self.k.device).view(1, 1, 1, D)
        lin = (blk[:, None, :, None] * (H * bs * D)          # [b, H, s, D] 线性下标
               + h_idx * (bs * D)
               + off[:, None, :, None] * D
               + d_idx)
        lay_k.view(-1)[lin.reshape(-1)] = k_new.reshape(-1)
        lay_v.view(-1)[lin.reshape(-1)] = v_new.reshape(-1)

    def read(self, layer_idx: int, tabs: torch.Tensor, read_len: int):
        """按 block 表把各请求的 cache gather 成连续 [b, kvh, read_len, hd]。
        gather 只搬需要的 block（nblk × 16 ≥ read_len），尾部 block 内多出的槽位
        由调用方的 attention mask 挡住。"""
        lay_k = self.k[layer_idx][tabs]                      # [b, nblk, kvh, bs, hd] gather 拷贝
        lay_v = self.v[layer_idx][tabs]
        b, nblk, H, bs, D = lay_k.shape
        k = lay_k.permute(0, 2, 1, 3, 4).reshape(b, H, nblk * bs, D)[:, :, :read_len]
        v = lay_v.permute(0, 2, 1, 3, 4).reshape(b, H, nblk * bs, D)[:, :, :read_len]
        return k, v
