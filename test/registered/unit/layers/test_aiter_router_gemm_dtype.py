import unittest

import torch

from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestAiterRouterGemmDtype(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.version.hip or not torch.cuda.is_available():
            raise unittest.SkipTest("requires a ROCm GPU and aiter")

    def test_router_logits_are_fp32_for_production_and_fallback_shapes(self):
        torch.manual_seed(20260912)

        # Cover the production GLM/DeepSeek router dimensions and an independent
        # non-multiple token count that takes aiter's fallback dispatch.
        for tokens, hidden_size, experts in ((8, 7168, 256), (65, 512, 64)):
            with self.subTest(
                tokens=tokens, hidden_size=hidden_size, experts=experts
            ):
                hidden_states = (
                    torch.randn(tokens, hidden_size, device="cuda")
                    / hidden_size**0.5
                ).to(torch.bfloat16)
                weight = torch.randn(
                    experts, hidden_size, device="cuda", dtype=torch.bfloat16
                )

                output = aiter_dsv3_router_gemm(hidden_states, weight)
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(output.shape, (tokens, experts))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
