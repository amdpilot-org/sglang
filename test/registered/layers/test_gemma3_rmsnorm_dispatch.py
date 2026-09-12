import unittest
from unittest import mock

import torch

from sglang.srt.layers import layernorm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGemma3RMSNormCudaDispatch(unittest.TestCase):
    def _layer(self, width=16, dtype=torch.float16):
        norm = layernorm.Gemma3RMSNorm(width).to(dtype=dtype)
        norm.weight.data.normal_(mean=0.0, std=0.1)
        return norm

    def test_high_rank_is_flattened_and_shape_is_restored(self):
        norm = self._layer()
        x = torch.randn(3, 4, 16, dtype=torch.float16)

        def kernel(flat, weight, eps):
            self.assertEqual(flat.shape, (12, 16))
            self.assertTrue(flat.is_contiguous())
            return norm.forward_native(flat)

        with mock.patch.object(
            layernorm, "gemma_rmsnorm", side_effect=kernel, create=True
        ) as op:
            out = norm.forward_cuda(x)

        op.assert_called_once()
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.allclose(out, norm.forward_native(x)))

    def test_noncontiguous_high_rank_is_materialized(self):
        norm = self._layer()
        wide = torch.randn(3, 4, 32, dtype=torch.float16)
        x = wide[..., :16]
        self.assertFalse(x.is_contiguous())

        def kernel(flat, weight, eps):
            self.assertTrue(flat.is_contiguous())
            return norm.forward_native(flat)

        with mock.patch.object(
            layernorm, "gemma_rmsnorm", side_effect=kernel, create=True
        ):
            out = norm.forward_cuda(x)

        self.assertTrue(torch.allclose(out, norm.forward_native(x)))

    def test_mixed_dtype_uses_native_path(self):
        norm = self._layer(dtype=torch.float32)
        x = torch.randn(5, 16, dtype=torch.bfloat16)

        with mock.patch.object(layernorm, "gemma_rmsnorm", create=True) as op:
            out = norm.forward_cuda(x)

        op.assert_not_called()
        self.assertTrue(torch.equal(out, norm.forward_native(x)))
        self.assertTrue(torch.isfinite(out).all())

    def test_mixed_dtype_residual_uses_native_path(self):
        norm = self._layer(dtype=torch.float32)
        x = torch.randn(5, 16, dtype=torch.bfloat16)
        residual = torch.randn_like(x)

        with mock.patch.object(layernorm, "gemma_fused_add_rmsnorm", create=True) as op:
            out, residual_out = norm.forward_cuda(x, residual)

        op.assert_not_called()
        expected, expected_residual = norm.forward_native(x, residual)
        self.assertTrue(torch.equal(out, expected))
        self.assertTrue(torch.equal(residual_out, expected_residual))

    def test_wrong_hidden_width_does_not_reach_kernel(self):
        norm = self._layer(width=16)
        x = torch.randn(2, 8, dtype=torch.float16)

        with mock.patch.object(layernorm, "gemma_rmsnorm", create=True) as op:
            with self.assertRaises(RuntimeError):
                norm.forward_cuda(x)

        op.assert_not_called()


if __name__ == "__main__":
    unittest.main()
