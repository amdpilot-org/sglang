import itertools
import unittest

import torch

from sglang.srt.layers.layernorm import RMSNorm


class TestRMSNormAMDRegression(unittest.TestCase):
    """Regression coverage for ROCm RMSNorm layout and residual semantics."""

    DTYPES = [torch.float16, torch.bfloat16]
    NUM_ROWS = [1, 7, 31]
    HIDDEN_SIZE = 128
    EPS = 1e-6

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise unittest.SkipTest("ROCm GPU is not available")
        torch.set_default_device("cuda")

    @staticmethod
    def _reference(x, weight, residual=None):
        accumulated = x.float()
        if residual is not None:
            accumulated = accumulated + residual.float()
        variance = accumulated.square().mean(dim=-1, keepdim=True)
        output = accumulated * torch.rsqrt(variance + TestRMSNormAMDRegression.EPS)
        output = (output * weight.float()).to(x.dtype)
        return output, accumulated.to(x.dtype)

    def test_plain_noncontiguous_input(self):
        for dtype, num_rows in itertools.product(self.DTYPES, self.NUM_ROWS):
            with self.subTest(dtype=dtype, num_rows=num_rows):
                torch.manual_seed(1000 + num_rows)
                layer = RMSNorm(self.HIDDEN_SIZE, eps=self.EPS).to(dtype=dtype)
                layer.weight.data.uniform_(0.5, 1.5)

                storage = torch.randn(num_rows, self.HIDDEN_SIZE * 2, dtype=dtype)
                x = storage[:, ::2]
                self.assertFalse(x.is_contiguous())
                x_before = x.clone()
                expected, _ = self._reference(x, layer.weight)

                with torch.inference_mode():
                    actual = layer(x)

                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
                torch.testing.assert_close(x, x_before, rtol=0, atol=0)

    def test_fused_residual_preserves_inputs(self):
        for dtype, num_rows in itertools.product(self.DTYPES, self.NUM_ROWS):
            with self.subTest(dtype=dtype, num_rows=num_rows):
                torch.manual_seed(2000 + num_rows)
                layer = RMSNorm(self.HIDDEN_SIZE, eps=self.EPS).to(dtype=dtype)
                layer.weight.data.uniform_(0.5, 1.5)
                x = torch.randn(num_rows, self.HIDDEN_SIZE, dtype=dtype)
                residual = torch.randn_like(x)
                x_before = x.clone()
                residual_before = residual.clone()
                expected, expected_residual = self._reference(
                    x, layer.weight, residual
                )

                with torch.inference_mode():
                    actual, actual_residual = layer(x, residual)

                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
                torch.testing.assert_close(
                    actual_residual, expected_residual, rtol=2e-2, atol=2e-3
                )
                torch.testing.assert_close(x, x_before, rtol=0, atol=0)
                torch.testing.assert_close(
                    residual, residual_before, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
