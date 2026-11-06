"""SwiGLU activation backend contracts; Triton cases run on a CUDA worker."""
import pathlib
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from layers import kernels
from layers.kernels import silu_and_mul


def _reference(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


class TestSiluAndMul(unittest.TestCase):
    def test_torch_backend_matches_reference(self):
        for shape in ((3, 256), (2, 3, 256), (2, 2050)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.float32)
                torch.testing.assert_close(silu_and_mul(x, backend="torch"), _reference(x))

    def test_auto_falls_back_for_cpu_and_noncontiguous_inputs(self):
        x = torch.randn(5, 128)
        torch.testing.assert_close(silu_and_mul(x), _reference(x))

        noncontiguous = torch.randn(128, 5).transpose(0, 1)
        self.assertFalse(noncontiguous.is_contiguous())
        torch.testing.assert_close(silu_and_mul(noncontiguous), _reference(noncontiguous))
        with self.assertRaisesRegex(ValueError, "contiguous CUDA"):
            silu_and_mul(noncontiguous, backend="triton")

    def test_auto_uses_torch_for_supported_cuda_input(self):
        if not torch.cuda.is_available():
            self.skipTest("backend selection check requires CUDA")
        x = torch.randn((8, 1536), device="cuda", dtype=torch.bfloat16)
        with mock.patch.object(kernels, "_silu_mul_kernel") as triton_kernel:
            actual = silu_and_mul(x, backend="auto")
        triton_kernel.assert_not_called()
        torch.testing.assert_close(actual, _reference(x))

    def test_auto_uses_triton_at_measured_large_token_threshold(self):
        if not torch.cuda.is_available():
            self.skipTest("backend selection check requires CUDA")
        if 'A800' not in torch.cuda.get_device_name():
            self.skipTest("crossover measurement covers A800 only")
        x = torch.randn((6144, 1536), device="cuda", dtype=torch.bfloat16)
        original_kernel = kernels._silu_mul_kernel
        launches = []

        class LaunchSpy:
            def __getitem__(self, grid):
                launch = original_kernel[grid]

                def run(*args, **kwargs):
                    launches.append(grid)
                    return launch(*args, **kwargs)

                return run

        with mock.patch.object(kernels, "_silu_mul_kernel", LaunchSpy()):
            actual = silu_and_mul(x, backend="auto")
        self.assertEqual(len(launches), 1)
        torch.testing.assert_close(actual, _reference(x), atol=2e-2, rtol=2e-2)

    def test_auto_keeps_unbenchmarked_cuda_dtype_on_torch(self):
        if not torch.cuda.is_available():
            self.skipTest("backend selection check requires CUDA")
        x = torch.randn((6144, 1536), device="cuda", dtype=torch.float16)
        with mock.patch.object(kernels, "_silu_mul_kernel") as triton_kernel:
            actual = silu_and_mul(x, backend="auto")
        triton_kernel.assert_not_called()
        torch.testing.assert_close(actual, _reference(x))

    def test_triton_backend_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("Triton kernel validation requires CUDA")
        cases = (
            (torch.float32, 2e-6),
            (torch.float16, 2e-3),
            (torch.bfloat16, 2e-2),
        )
        for dtype, tol in cases:
            for shape in ((1, 2), (7, 254), (2, 3, 2050)):
                with self.subTest(dtype=dtype, shape=shape):
                    torch.manual_seed(731)
                    x = torch.randn(shape, device="cuda", dtype=dtype)
                    actual = silu_and_mul(x, backend="triton")
                    torch.testing.assert_close(actual, _reference(x), atol=tol, rtol=tol)

    def test_empty_shape_does_not_launch(self):
        x = torch.empty((0, 128), dtype=torch.float32)
        self.assertEqual(silu_and_mul(x).shape, (0, 64))

    def test_odd_swiglu_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "even final dimension"):
            silu_and_mul(torch.randn(3, 127))


if __name__ == "__main__":
    unittest.main(verbosity=2)
