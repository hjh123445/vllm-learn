"""Check EP combine preserves the requested expert accumulation policy."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layers.moe import _reduce_routes


class RouteReductionTest(unittest.TestCase):
    def test_expert_order_and_rounding(self):
        routes = torch.tensor([[[256.], [1.], [-256.]]], dtype=torch.bfloat16)
        experts = torch.tensor([[9, 2, 7]])
        with patch.dict(os.environ, VLLM_LEARN_MOE_REDUCTION='hf_reference'):
            # Ascending experts: 1 + (-256) + 256; original order gives 0.
            expected = ((routes[:, 1] + routes[:, 2]) + routes[:, 0])
            torch.testing.assert_close(_reduce_routes(routes, experts), expected,
                                       atol=0, rtol=0)
        with patch.dict(os.environ, VLLM_LEARN_MOE_REDUCTION='fp32'):
            torch.testing.assert_close(_reduce_routes(routes, experts),
                                       torch.ones((1, 1), dtype=routes.dtype), atol=0, rtol=0)

    def test_empty_origins(self):
        for policy in ('fp32', 'hf_reference'):
            with patch.dict(os.environ, VLLM_LEARN_MOE_REDUCTION=policy):
                actual = _reduce_routes(torch.empty(0, 8, 32), torch.empty(0, 8, dtype=torch.long))
                self.assertEqual(actual.shape, (0, 32))


if __name__ == '__main__':
    unittest.main()
