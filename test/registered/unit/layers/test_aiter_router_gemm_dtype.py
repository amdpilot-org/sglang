import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestAiterRouterGemmDtype(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.version.hip or not torch.cuda.is_available():
            raise unittest.SkipTest("requires a ROCm GPU and aiter")

    def test_router_logits_retain_fp32_precision(self):
        # Production dimensions plus independent fallback and skinny boundaries.
        for tokens, hidden_size, experts in (
            (8, 7168, 256),
            (65, 512, 64),
            (1, 513, 63),
        ):
            with self.subTest(
                tokens=tokens, hidden_size=hidden_size, experts=experts
            ):
                torch.manual_seed(20260912 + tokens)
                hidden_states = (
                    torch.randn(tokens, hidden_size, device="cuda")
                    / hidden_size**0.5
                ).to(torch.bfloat16)
                weight = torch.randn(
                    experts, hidden_size, device="cuda", dtype=torch.bfloat16
                )

                output = aiter_dsv3_router_gemm(hidden_states, weight)
                reference = F.linear(hidden_states.float(), weight.float())
                rounded_reference = reference.to(torch.bfloat16).float()

                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(output.shape, (tokens, experts))
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreater(
                    torch.count_nonzero(output != output.to(torch.bfloat16).float()),
                    0,
                    "router logits were computed in bf16 and only widened to fp32",
                )
                self.assertLessEqual(
                    (output - reference).abs().max(),
                    (rounded_reference - reference).abs().max(),
                )


if __name__ == "__main__":
    unittest.main()
