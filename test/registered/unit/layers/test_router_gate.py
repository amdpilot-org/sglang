import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.router_gate import RouterGate, router_linear_bf16_fp32
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRouterGate(unittest.TestCase):
    def test_fp32_reference_and_output_contract(self):
        torch.manual_seed(7)
        gate = RouterGate(17, 9, fp32_compute=False, params_dtype=torch.bfloat16)
        x = torch.randn(8, 17, dtype=torch.bfloat16)
        with torch.no_grad():
            gate.weight.copy_(torch.randn_like(gate.weight))
        actual = gate(x)
        reference = x.float() @ gate.weight.float().t()
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_fp32_compute_contract(self):
        gate = RouterGate(5, 3, fp32_compute=True, has_correction_bias=True)
        self.assertEqual(gate.weight.dtype, torch.float32)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)
        self.assertEqual(
            gate(torch.randn(2, 5, dtype=torch.bfloat16)).dtype, torch.float32
        )

    def test_deterministic_policy_skips_tiny_dispatch(self):
        x = torch.randn(1, 4, dtype=torch.bfloat16)
        weight = torch.randn(3, 4, dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.moe.router_gate.envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get",
            return_value=True,
        ):
            actual = router_linear_bf16_fp32(x, weight, tiny_max_tokens=16)
        torch.testing.assert_close(
            actual, x.float() @ weight.float().t(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
