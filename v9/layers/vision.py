"""Qwen3-Omni vision tower 自研实现(教学版,与 HF Qwen3OmniMoeVisionEncoder 语义对齐)。

组件(对照 HF transformers models/qwen3_omni_moe):
  - PatchEmbed      : Conv3d(3→1152, kernel/stride=(2,16,16)),patch 线性化
  - PosEmbedInterp  : 48×48 学习表 → 双线性(align_corners=True, border)重采样到任意
                      (h,w) 网格,2×2 merge-block-major 顺序(closed-form taps+weights,
                      对齐 transformers.vision_utils.get_vision_interpolation_*)
  - VisionRope      : axial 2D rotary(h/w 两轴,head_dim=72,theta=10000),与文本 rope
                      同数学(freq 复制两半 = engine cos|sin 缓存格式)
  - Block           : LN(1e-6)+全量图像内注意力(SDPA,无 mask,per-image varlen)+
                      LN+gelu_tanh MLP
  - Merger          : LN → [2×2 邻域 concat=4608] → Linear→GELU→Linear(→2048)
                      deepstack merger 用 postshuffle norm(4608 维 LN),主 merger 用
                      pre-shuffle norm(1152 维 LN)
接口:
  tower = Qwen3VisionTower(cfg_dict)          # cfg_dict = thinker_config['vision_config']
  embeds, deepstack = tower(pixel_values, grid_thw)
    pixel_values : [sum(t*h*w), in_channels*temporal*patch*patch](预处理后的 patch 行)
    grid_thw     : [num_images, 3] int64
    embeds       : [sum(t*(h/2)*(w/2)), 2048] 浮点与权重同 dtype
    deepstack[i] : [同形状] ×3(deepstack_visual_indexes 每层的特征)
权重加载:load_vision_state_dict 接受 HF 前缀 thinker.visual.* 的映射。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- rotary(axial 2D,与 HF Qwen3OmniMoeVisionRotaryEmbedding 一致) ----------

_VISION_ROPE_CACHE: dict = {}


def vision_rope_cache(head_dim: int, base: float, device) -> torch.Tensor:
    """[S_max?, head_dim] cos|sin 缓存按 pos 索引现查:返回 inv 频率,用时现算更省存。
    这里存 inv [head_dim//2],forward 内 cos/sin 现算(fp32)。"""
    key = (head_dim, base, str(device))
    if key not in _VISION_ROPE_CACHE:
        half = head_dim // 2
        inv = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
        _VISION_ROPE_CACHE[key] = inv.to(device)
    return _VISION_ROPE_CACHE[key]


def apply_vision_rope(x: torch.Tensor, inv: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
    """x [S,H,D](被调用方已转 float);pos_ids [2,S](h 轴,w 轴)。
    inv [D/4](= HF VisionRotaryEmbedding(dim=D/2) 的 inv_freq,即 theta^{-j/(D/2)})。
    每轴 18 个频率 → [fh,fw] 36 段;再按 HF 的 cat((f,f)) 语义重复 → [S,72]。"""
    fh = pos_ids[0].float().unsqueeze(-1) * inv  # [S,18]
    fw = pos_ids[1].float().unsqueeze(-1) * inv  # [S,18]
    # emb36 = [fh,fw](HF emb = cat([fh,fw,fh,fw]) 的前半,rotate-half 只需这一段)
    emb36 = torch.cat([fh, fw], dim=-1).unsqueeze(1)  # [S,1,36]
    cos, sin = emb36.cos(), emb36.sin()
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(x.dtype)


# ---------- 位置嵌入插值 + block-major 位置 id(对齐 hf vision_utils) ----------

def vision_interp(grid_thw: torch.Tensor, merge: int, side: int, total: int | None = None):
    """per-patch 双线性 taps(允许逐 patch 尺寸不同:src 用 patch 所属图的 h/w)。"""
    device = grid_thw.device
    t_, h_, w_ = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    counts = t_ * h_ * w_
    if total is None:
        total = int(counts.sum())
    heights = torch.repeat_interleave(h_, counts, output_size=total)
    widths = torch.repeat_interleave(w_, counts, output_size=total)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts, output_size=total)
    within = (torch.arange(total, device=device) - starts) % (heights * widths)
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col

    def taps(idx, sizes):
        src = idx.float() * (side - 1) / (sizes.float() - 1).clamp(min=1)
        floor = torch.floor(src)
        t2 = torch.stack([floor, floor + 1], dim=1).long().clamp(0, side - 1)
        d = (src.unsqueeze(1) - floor.unsqueeze(1) - torch.arange(2, device=device)).abs()
        w = (1 - d).clamp(min=0)
        return t2, w

    ht, hwt = taps(row, heights)
    wt, wwt = taps(col, widths)
    indices = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
    weights = (hwt[:, :, None] * wwt[:, None, :]).reshape(-1, 4)
    return indices, weights


def vision_position_ids(grid_thw: torch.Tensor, merge: int, grid_rows=None) -> torch.Tensor:
    """[total, 2] block-major 的 (h,w) 位置 id(与 HF get_vision_position_ids 一致,merge=2)。"""
    ids = []
    for t, h, w in (grid_thw.tolist() if grid_rows is None else grid_rows):
        hp, wp = torch.meshgrid(
            torch.arange(h, device=grid_thw.device),
            torch.arange(w, device=grid_thw.device),
            indexing='ij')
        block = (h // merge, merge, w // merge, merge)
        hp = hp.reshape(block).permute(0, 2, 1, 3).flatten()
        wp = wp.reshape(block).permute(0, 2, 1, 3).flatten()
        ids.append(torch.stack([hp, wp], dim=-1).repeat(t, 1))
    return torch.cat(ids, dim=0)


# ---------- 模块 ----------

class Qwen3VisionAttention(nn.Module):
    """全量图像内注意力(无 mask),qkv 有 bias;per-image varlen SDPA 循环(教学版)。"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cu: torch.Tensor, inv: torch.Tensor,
                pos: torch.Tensor) -> torch.Tensor:
        s = x.size(0)
        q, k, v = (self.qkv(x).reshape(s, 3, self.num_heads, self.head_dim)
                   .permute(1, 0, 2, 3).unbind(0))
        q = apply_vision_rope(q.float(), inv, pos).to(x.dtype)
        k = apply_vision_rope(k.float(), inv, pos).to(x.dtype)
        out = torch.empty_like(q)
        cu_list = cu.tolist() if isinstance(cu, torch.Tensor) else cu
        for st, en in zip(cu_list[:-1], cu_list[1:]):
            if en > st:
                out[st:en] = F.scaled_dot_product_attention(
                    q[st:en].transpose(0, 1).unsqueeze(0),
                    k[st:en].transpose(0, 1).unsqueeze(0),
                    v[st:en].transpose(0, 1).unsqueeze(0))[0].transpose(0, 1)
        return self.proj(out.reshape(s, self.dim))


class Qwen3VisionMLP(nn.Module):
    def __init__(self, dim: int, inter: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, inter, bias=True)
        self.linear_fc2 = nn.Linear(inter, dim, bias=True)
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3VisionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, inter: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Qwen3VisionAttention(dim, num_heads)
        self.mlp = Qwen3VisionMLP(dim, inter)

    def forward(self, x, cu, inv, pos):
        x = x + self.attn(self.norm1(x), cu, inv, pos)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VisionPatchMerger(nn.Module):
    """use_postshuffle_norm=True(deepstack):LN 作用在 2×2 拼接后的 4608 维;
    False(主 merger):LN 作用在 1152 维,再 reshape 成 4608。"""

    def __init__(self, dim: int, merge: int, out_dim: int, postshuffle: bool):
        super().__init__()
        self.hidden_size = dim * merge * merge
        self.postshuffle = postshuffle
        self.ln_q = nn.LayerNorm(self.hidden_size if postshuffle else dim, eps=1e-6)
        self.mlp = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, out_dim),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x.view(-1, self.hidden_size) if self.postshuffle else x).view(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class Qwen3VisionTower(nn.Module):
    """patch_embed → +插值 pos_embed → 27×block → merger(+3 路 deepstack merger)。"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg['hidden_size']
        self.depth = cfg['depth']
        self.merge = cfg['spatial_merge_size']
        self.deepstack_idx = cfg['deepstack_visual_indexes']
        self.side = int(round(cfg['num_position_embeddings'] ** 0.5))
        self.rope_base = 10000.0

        self.patch_embed = nn.Conv3d(
            cfg['in_channels'], self.dim,
            kernel_size=[cfg['temporal_patch_size'], cfg['patch_size'], cfg['patch_size']],
            stride=[cfg['temporal_patch_size'], cfg['patch_size'], cfg['patch_size']],
            bias=True)
        self.pos_embed = nn.Embedding(cfg['num_position_embeddings'], self.dim)
        self.blocks = nn.ModuleList([
            Qwen3VisionBlock(self.dim, cfg['num_heads'], cfg['intermediate_size'])
            for _ in range(self.depth)])
        self.merger = Qwen3VisionPatchMerger(self.dim, self.merge, cfg['out_hidden_size'], postshuffle=False)
        self.mergers_ds = nn.ModuleList([
            Qwen3VisionPatchMerger(self.dim, self.merge, cfg['out_hidden_size'], postshuffle=True)
            for _ in self.deepstack_idx])

    def forward(self, pixel: torch.Tensor, grid_thw: torch.Tensor):
        device = pixel.device
        target_dtype = self.patch_embed.weight.dtype
        total = pixel.size(0)
        # Grid metadata is known on the host before the pixel H2D copy.
        # Read it once, never once per vision layer/attention invocation.
        grid_rows = grid_thw.detach().cpu().tolist()
        if (not grid_rows or any(min(t, h, w) <= 0 or h % self.merge or w % self.merge
                                 for t, h, w in grid_rows)
                or sum(t * h * w for t, h, w in grid_rows) != total):
            raise ValueError('complete pixel rows must match positive, merge-aligned image grids')
        grid_device = grid_thw.to(device)
        x = pixel.view(total, self.cfg['in_channels'], self.cfg['temporal_patch_size'],
                       self.cfg['patch_size'], self.cfg['patch_size'])
        x = self.patch_embed(x.to(target_dtype)).view(total, self.dim)

        interp_idx, interp_w = vision_interp(grid_device, self.merge, self.side, total=total)
        pos_embeds = (self.pos_embed(interp_idx) * interp_w[:, :, None]).sum(1)
        x = x + pos_embeds.to(x.dtype)

        pos_ids = vision_position_ids(grid_device, self.merge, grid_rows=grid_rows).t().contiguous()
        # HF vision rope:inv_freq = theta^{-arange(0,D/2,2)/(D/2)}(D=head_dim)
        head_dim = self.dim // self.cfg['num_heads']
        inv = vision_rope_cache(head_dim // 2, self.rope_base, device)
        # HF attends independently within each temporal slice of each grid.
        # For still images t=1; retaining host offsets also avoids 27 D2H
        # synchronizations from cu.tolist() in the block loop.
        cu = [0]
        for t, h, w in grid_rows:
            for _ in range(t):
                cu.append(cu[-1] + h * w)

        deep = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, cu, inv, pos_ids)
            if i in self.deepstack_idx:
                deep.append(self.mergers_ds[self.deepstack_idx.index(i)](x))
        main = self.merger(x)
        return main, deep

    @torch.no_grad()
    def load_vision_state_dict(self, sd: dict, prefix: str = 'thinker.visual.'):
        """从 HF 命名的 state dict 子集加载(缺失会报错)。"""
        def g(name):
            k = prefix + name
            if k not in sd:
                raise KeyError(k)
            return sd[k]

        def farm(mod: nn.Module, name: str):
            if isinstance(mod, nn.Linear):
                mod.weight.copy_(g(name + '.weight'))
                if mod.bias is not None:
                    mod.bias.copy_(g(name + '.bias'))
            elif isinstance(mod, nn.LayerNorm):
                mod.weight.copy_(g(name + '.weight'))
                mod.bias.copy_(g(name + '.bias'))
            elif isinstance(mod, nn.Conv3d):
                mod.weight.copy_(g(name + '.weight'))
                mod.bias.copy_(g(name + '.bias'))
            elif isinstance(mod, nn.Embedding):
                mod.weight.copy_(g(name + '.weight'))

        farm(self.patch_embed, 'patch_embed.proj')
        farm(self.pos_embed, 'pos_embed')
        for i, blk in enumerate(self.blocks):
            farm(blk.norm1, f'blocks.{i}.norm1')
            farm(blk.norm2, f'blocks.{i}.norm2')
            farm(blk.attn.qkv, f'blocks.{i}.attn.qkv')
            farm(blk.attn.proj, f'blocks.{i}.attn.proj')
            farm(blk.mlp.linear_fc1, f'blocks.{i}.mlp.linear_fc1')
            farm(blk.mlp.linear_fc2, f'blocks.{i}.mlp.linear_fc2')
        for wname, mod in (('merger', self.merger), *[(f'merger_list.{i}', m) for i, m in enumerate(self.mergers_ds)]):
            farm(mod.ln_q, wname + '.ln_q')
            for j, lin in enumerate([l for l in mod.mlp if isinstance(l, nn.Linear)]):
                farm(lin, wname + f'.mlp.{j * 2}')
