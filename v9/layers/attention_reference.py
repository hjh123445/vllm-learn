"""Slow paged SDPA oracle for numerical diagnosis, excluded from benchmarks."""
import os
from pathlib import Path

import torch
import torch.nn.functional as F


_debug_capture_count = 0


def paged_sdpa(q, k_cache, v_cache, block_tables, query_lengths, key_lengths, scale,
               *, masked_gqa=None):
    """Match right-aligned causal attention, including cached chunked prefill.

    Lengths are host lists. Gathering KV and iterating requests is intentionally
    explicit: this path checks semantics and is not a serving optimization.
    """
    if masked_gqa is None:
        masked_gqa = os.getenv('VLLM_LEARN_SDPA_MASKED_GQA', '0') == '1'
    output = torch.empty_like(q)
    block_size, offset = k_cache.shape[1], 0
    for seq, (nq, nk) in enumerate(zip(query_lengths, key_lengths)):
        if nq == 0:
            continue
        if not 0 < nq <= nk:
            raise ValueError('SDPA oracle requires 0 < query length <= key length')
        positions = torch.arange(nk, device=q.device)
        blocks = block_tables[seq, positions // block_size].long()
        key = k_cache[blocks, positions % block_size].transpose(0, 1).unsqueeze(0)
        value = v_cache[blocks, positions % block_size].transpose(0, 1).unsqueeze(0)
        query = q[offset:offset+nq].transpose(0, 1).unsqueeze(0)
        mask = None
        causal = nq == nk and nq > 1
        if 1 < nq < nk:
            # SDPA's is_causal=True is top-left aligned for non-square inputs.
            absolute_q = torch.arange(nk-nq, nk, device=q.device)
            mask = positions[None, :] <= absolute_q[:, None]
        grouped = query.shape[1] != key.shape[1]
        # Match HF's CUDA SDPA policy: a masked GQA call can select a different
        # backend from explicit KV repetition. Keep the old policy for diagnosis.
        use_gqa = grouped and (masked_gqa or (mask is None and key.shape[-1] <= 256))
        if grouped and not use_gqa:
            repeats = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        result = F.scaled_dot_product_attention(query, key, value,
            attn_mask=mask, is_causal=causal, dropout_p=0., scale=scale,
            enable_gqa=use_gqa)
        capture_dir = os.getenv('VLLM_LEARN_SDPA_CAPTURE_DIR')
        target_k = int(os.getenv('VLLM_LEARN_SDPA_CAPTURE_KEY_LENGTH', '-1'))
        max_captures = int(os.getenv('VLLM_LEARN_SDPA_CAPTURE_MAX', '8'))
        global _debug_capture_count
        if (capture_dir and _debug_capture_count < max_captures and seq == 0
                and nq == 1 and nk == target_k):
            rank = (torch.distributed.get_rank()
                    if torch.distributed.is_available()
                    and torch.distributed.is_initialized() else 0)
            if rank == 0:
                Path(capture_dir).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'query': query.detach().cpu(), 'key': key.detach().cpu(),
                    'value': value.detach().cpu(), 'result': result.detach().cpu(),
                    'mask': mask.detach().cpu() if mask is not None else None,
                    'causal': causal, 'scale': scale, 'use_gqa': use_gqa,
                    'seq': seq, 'query_length': nq, 'key_length': nk,
                }, Path(capture_dir) / f'layer{_debug_capture_count}.pt')
            _debug_capture_count += 1
        output[offset:offset+nq] = result[0].transpose(0, 1)
        offset += nq
    if offset != q.shape[0]:
        raise ValueError('SDPA oracle query lengths do not cover the query tensor')
    return output.flatten(1)
