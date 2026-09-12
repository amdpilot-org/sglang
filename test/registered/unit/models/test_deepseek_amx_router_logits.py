"""Regression tests for the DeepSeek Intel AMX router-logit contract."""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.models.deepseek_v2 import MoEGate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekAmxRouterLogits(CustomTestCase):
    def test_cpu_bf16_weight_emits_fp32_logits_without_packed_linear(self):
        """CPU must not round the fp32 accumulator through the AMX BF16 output."""
        gate = MoEGate.__new__(MoEGate)
        torch.nn.Module.__init__(gate)
        gate.weight = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16),
            requires_grad=False,
        )
        hidden_states = torch.tensor([[1.001, 1.002]], dtype=torch.bfloat16)

        with (
            patch.object(deepseek_v2, "_is_cuda", False),
            patch.object(
                torch.ops.sgl_kernel,
                "weight_packed_linear",
                create=True,
            ) as packed_linear,
        ):
            logits = gate(hidden_states)

        packed_linear.assert_not_called()
        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(
            logits,
            torch.nn.functional.linear(hidden_states.float(), gate.weight.float()),
        )

    def test_fp32_router_weight_boundary_remains_fp32(self):
        """The existing non-AMX fallback is the semantic boundary case."""
        gate = MoEGate.__new__(MoEGate)
        torch.nn.Module.__init__(gate)
        gate.weight = torch.nn.Parameter(
            torch.eye(2, dtype=torch.float32), requires_grad=False
        )
        hidden_states = torch.tensor([[1.001, 1.002]], dtype=torch.float32)

        with patch.object(deepseek_v2, "_is_cuda", False):
            logits = gate(hidden_states.to(torch.bfloat16))

        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(
            logits,
            torch.nn.functional.linear(
                hidden_states.to(torch.bfloat16).float(), gate.weight
            ),
        )

    def test_bf16_quantization_can_erase_router_ordering(self):
        """Casting after the gate cannot recover distinctions already rounded away."""
        fp32_logits = torch.tensor([[1.001, 1.002]], dtype=torch.float32)
        bf16_roundtrip = fp32_logits.to(torch.bfloat16).float()

        self.assertGreater(fp32_logits[0, 1], fp32_logits[0, 0])
        self.assertEqual(bf16_roundtrip[0, 1], bf16_roundtrip[0, 0])


if __name__ == "__main__":
    unittest.main()
