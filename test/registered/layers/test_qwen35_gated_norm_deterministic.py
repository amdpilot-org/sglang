"""Regression coverage for Qwen3.5 deterministic GDN normalization.

Qwen3.5-27B uses 48 value heads of width 128.  The same token must not
change numerically when scheduler batching changes the total number of rows
passed to the fused gated RMSNorm kernel.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import (
    MAX_ROWS_PER_BLOCK,
    calc_rows_per_block,
    rms_norm_gated,
    rms_norm_ref,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=12, stage="stage-b", runner_config="1-gpu-large-amd")


class TestQwen35GatedNormDeterministic(unittest.TestCase):
    @patch(
        "sglang.kernels.ops.attention.fla.layernorm_gated."
        "is_batch_invariant_mode_enabled",
        return_value=True,
    )
    def test_launch_geometry_is_independent_of_batch_rows(self, _):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Include boundaries around every power of two used by the adaptive
        # policy and odd sizes representative of dynamic scheduler batches.
        for rows in (1, 2, 3, 31, 32, 33, 255, 256, 257, 1025, 4097):
            with self.subTest(rows=rows):
                self.assertEqual(calc_rows_per_block(rows, device), MAX_ROWS_PER_BLOCK)

    @unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
    @patch(
        "sglang.kernels.ops.attention.fla.layernorm_gated."
        "is_batch_invariant_mode_enabled",
        return_value=True,
    )
    def test_qwen35_27b_row_is_bitwise_invariant_and_matches_reference(self, _):
        torch.manual_seed(31261)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        head_dim = 128
        large_batch_rows = 1025

        x_single = torch.randn(1, head_dim, device=device, dtype=dtype)
        z_single = torch.randn_like(x_single)
        weight = torch.randn(head_dim, device=device, dtype=dtype)

        x_batch = torch.randn(large_batch_rows, head_dim, device=device, dtype=dtype)
        z_batch = torch.randn_like(x_batch)
        x_batch[0].copy_(x_single[0])
        z_batch[0].copy_(z_single[0])

        actual_single = rms_norm_gated(
            x=x_single,
            weight=weight,
            bias=None,
            z=z_single,
            eps=1e-6,
            is_rms_norm=True,
        )
        actual_batched = rms_norm_gated(
            x=x_batch,
            weight=weight,
            bias=None,
            z=z_batch,
            eps=1e-6,
            is_rms_norm=True,
        )
        expected = rms_norm_ref(
            x_single,
            weight,
            None,
            z_single,
            eps=1e-6,
        )

        self.assertTrue(torch.equal(actual_single[0], actual_batched[0]))
        torch.testing.assert_close(
            actual_single.float(), expected.float(), atol=2e-2, rtol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
