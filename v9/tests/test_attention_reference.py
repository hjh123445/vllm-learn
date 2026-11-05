"""The diagnostic oracle must preserve chunk offsets, GQA and page order."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layers.attention_reference import paged_sdpa


class PagedSDPATest(unittest.TestCase):
    def test_chunks_decode_and_gqa_with_shuffled_pages(self):
        torch.manual_seed(42)
        bs, hq, hk, hd = 4, 4, 2, 8
        kc, vc = torch.randn(5, bs, hk, hd), torch.randn(5, bs, hk, hd)
        tables = torch.tensor([[3, 0, 4], [2, 1, 0]])
        for qlens, klens in (([3, 5], [3, 5]), ([2, 3], [7, 9]), ([1, 1], [7, 9])):
            q = torch.randn(sum(qlens), hq, hd)
            expected, offset = [], 0
            for seq, (nq, nk) in enumerate(zip(qlens, klens)):
                pos = torch.arange(nk)
                key = kc[tables[seq, pos//bs], pos%bs].repeat_interleave(hq//hk, 1)
                value = vc[tables[seq, pos//bs], pos%bs].repeat_interleave(hq//hk, 1)
                score = torch.einsum('qhd,khd->hqk', q[offset:offset+nq], key)*hd**-.5
                mask = pos[None, :] > torch.arange(nk-nq, nk)[:, None]
                prob = score.masked_fill(mask, -torch.inf).softmax(-1)
                expected.append(torch.einsum('hqk,khd->qhd', prob, value).flatten(1))
                offset += nq
            for masked_gqa in (False, True):
                actual = paged_sdpa(q, kc, vc, tables, qlens, klens, hd**-.5,
                                    masked_gqa=masked_gqa)
                torch.testing.assert_close(actual, torch.cat(expected), atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
