"""Multimodal image preprocessing for the Qwen3-Omni vision tower.

Semantics match the HF qwen3_omni image processor:
  - smart_resize (factor = patch*merge = 32, min/max pixel clamps)
  - normalize with mean=std=0.5 after /255 rescale
  - temporal pad (repeat last frame) to temporal_patch=2
  - block-major (2x2 spatial-merge) patch unfolding: one row per patch with
    layout [C, temporal, patch, patch] flattened (1176 dims)
Returns (patches [t*h*w, 1176] fp32, grid_thw [1,3] int64).
Unit test compares against AutoProcessor pixel_values/image_grid_thw.
"""
from __future__ import annotations

import numpy as np
import torch

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def smart_resize(height: int, width: int, factor: int = 32,
                 min_pixels: int = 3136, max_pixels: int = 12845056) -> tuple[int, int]:
    """Exact port of HF smart_resize (round then clamp min/max by factor)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError('aspect ratio beyond 200:1 not supported')
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = (height * width / max_pixels) ** 0.5
        h_bar = max(factor, int(np.floor(height / beta / factor) * factor))
        w_bar = max(factor, int(np.floor(width / beta / factor) * factor))
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        h_bar = max(factor, int(np.ceil(height * beta / factor) * factor))
        w_bar = max(factor, int(np.ceil(width * beta / factor) * factor))
    return h_bar, w_bar


def image_to_patches(img, patch: int = 16, temporal: int = 2, merge: int = 2,
                     min_pixels: int = 3136, max_pixels: int = 12845056):
    """img: [H,W,3] uint8 numpy or PIL image or path.
    Returns (patches [t*h*w, C*temporal*patch*patch] fp32, grid [1,3] int64)."""
    import PIL.Image
    if isinstance(img, str):
        img = np.asarray(PIL.Image.open(img).convert('RGB'))
    elif not isinstance(img, np.ndarray):
        img = np.asarray(img.convert('RGB') if hasattr(img, 'convert') else img)
    h, w = img.shape[:2]
    hr, wr = smart_resize(h, w, patch * merge, min_pixels, max_pixels)
    if (hr, wr) != (h, w):
        pil = PIL.Image.fromarray(img)
        img = np.asarray(pil.resize((wr, hr), PIL.Image.BICUBIC))
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD                        # [H',W',3]
    x = torch.from_numpy(x).permute(2, 0, 1)    # [3,H',W']
    x = x.unsqueeze(0)                          # [1,3,H',W']
    if x.size(0) % temporal != 0:
        pad = temporal - x.size(0) % temporal
        x = torch.cat([x, x[-1:].repeat(pad, 1, 1, 1)], dim=0)
    t = x.size(0) // temporal
    grid_h, grid_w = hr // patch, wr // patch
    # [t, temporal, C, gh//m, m, patch, gw//m, m, patch] -> block-major permute
    x = x.view(t, temporal, 3, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)    # t, gh//m, gw//m, m, m, C, temporal, p, p
    patches = x.reshape(t * grid_h * grid_w, 3 * temporal * patch * patch)
    grid = torch.tensor([[t, grid_h, grid_w]], dtype=torch.int64)
    return patches, grid


def mrope_positions(prompt_len: int, images) -> tuple[np.ndarray, int]:
    """HF get_rope_index(仅图像部分)的闭式复刻:
    text 段三轴均匀递增;图像段 t 轴常数,h/w 走 llm 网格(h//merge, w//merge,
    raster t→h→w);每段结束 st_pos = 段内 max + 1(三轴一起)。
    images: [(pos_in_prompt, t_llm, gh_llm, gw_llm)],按 pos 升序。
    返回 (pos3 int32 [3, prompt_len], delta int)。
      delta = max(pos3)+1 - prompt_len —— decode 第 k 个 token 的标量位置
      = (len-1+k) + delta(此时三轴相等,可退化为普通 rope)。"""
    pos3 = np.zeros((3, prompt_len), dtype=np.int64)
    off = 0
    cur = 0
    for pos, t, gh, gw in sorted(images, key=lambda x: x[0]):
        n = t * gh * gw
        text_n = pos - cur
        pos3[:, cur:pos] = np.arange(off, off + text_n)[None, :]
        off += text_n
        t_idx = np.arange(t)[:, None, None] + np.zeros((1, gh, gw), dtype=np.int64)
        h_idx = np.arange(gh)[None, :, None] + np.zeros((t, 1, gw), dtype=np.int64)
        w_idx = np.arange(gw)[None, None, :] + np.zeros((t, gh, 1), dtype=np.int64)
        blk = np.stack([t_idx.reshape(-1), h_idx.reshape(-1), w_idx.reshape(-1)], 0) + off
        pos3[:, pos:pos + n] = blk
        off = int(blk.max()) + 1
        cur = pos + n
    tail = prompt_len - cur
    pos3[:, cur:] = np.arange(off, off + tail)[None, :]
    off += tail
    delta = off - prompt_len
    return pos3.astype(np.int32), int(delta)
