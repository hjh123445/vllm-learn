"""The quantized linear must use its reference path outside Triton constraints."""
import pathlib
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from layers.quantized import QuantizedLinear, dequantize


class TestQuantizedFallback(unittest.TestCase):
    def test_cpu_quantized_linear_falls_back_from_triton_decode_kernel(self):
        torch.manual_seed(29)
        weight = torch.randn(32, 128, dtype=torch.bfloat16)
        module = QuantizedLinear(128, 32, group=64)
        module.weight_loader(None, weight)
        x = torch.randn(4, 128, dtype=torch.float32)

        actual = module(x)
        dequantized = dequantize(module.w_packed, module.w_scales, module.group)
        expected = F.linear(x, dequantized)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
