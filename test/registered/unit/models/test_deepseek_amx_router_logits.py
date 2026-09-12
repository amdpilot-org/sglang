"""Regression tests for the DeepSeek Intel AMX router-logit contract."""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v2 as deepseek_v2
import sglang.srt.models.ernie4 as ernie4
from sglang.srt.layers.moe.topk import biased_grouped_topk_cpu
from sglang.srt.models.deepseek_v2 import MoEGate
from sglang.srt.models.ernie4 import MoEGate as Ernie4MoEGate
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
            patch.object(deepseek_v2, "use_intel_amx_backend", return_value=True),
            patch.object(
                torch.ops.sgl_kernel,
                "weight_packed_linear",
                create=True,
                return_value=torch.tensor(
                    [[1.0, 1.0]], dtype=torch.bfloat16
                ),
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

    def test_cpu_topk_rejects_bf16_router_logits_before_native_dispatch(self):
        """The CPU TopK boundary must not silently accept pre-rounded logits."""
        hidden_states = torch.zeros((1, 2), dtype=torch.bfloat16)
        gating_output = torch.tensor([[1.001, 1.002]], dtype=torch.bfloat16)
        correction_bias = torch.zeros(2, dtype=torch.float32)

        with patch.object(
            torch.ops.sgl_kernel, "biased_grouped_topk_cpu", create=True
        ) as native_topk:
            with self.assertRaisesRegex(ValueError, "requires FP32 gating_output"):
                biased_grouped_topk_cpu(
                    hidden_states,
                    gating_output,
                    correction_bias,
                    topk=1,
                    renormalize=False,
                    num_expert_group=1,
                    topk_group=1,
                )

        native_topk.assert_not_called()

    def test_cpu_topk_forwards_fp32_router_logits(self):
        """FP32 router logits remain accepted and reach the native kernel unchanged."""
        hidden_states = torch.zeros((1, 2), dtype=torch.bfloat16)
        gating_output = torch.tensor([[1.001, 1.002]], dtype=torch.float32)
        correction_bias = torch.zeros(2, dtype=torch.float32)
        expected = (
            torch.ones((1, 1), dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.int32),
        )

        with patch.object(
            torch.ops.sgl_kernel,
            "biased_grouped_topk_cpu",
            create=True,
            return_value=expected,
        ) as native_topk:
            actual = biased_grouped_topk_cpu(
                hidden_states,
                gating_output,
                correction_bias,
                topk=1,
                renormalize=False,
                num_expert_group=1,
                topk_group=1,
            )

        self.assertIs(actual, expected)
        self.assertIs(native_topk.call_args.args[1], gating_output)


class TestErnie4CpuRouterLogits(CustomTestCase):
    def _gate(self, dtype):
        gate = Ernie4MoEGate.__new__(Ernie4MoEGate)
        torch.nn.Module.__init__(gate)
        gate.weight = torch.nn.Parameter(
            torch.eye(2, dtype=dtype), requires_grad=False
        )
        return gate

    def test_cpu_bf16_weight_emits_fp32_logits_accepted_by_topk(self):
        gate = self._gate(torch.bfloat16)
        hidden_states = torch.tensor([[1.001, 1.002]], dtype=torch.bfloat16)

        with patch.object(ernie4, "_is_cpu", True):
            logits = gate(hidden_states)

        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(
            logits,
            torch.nn.functional.linear(hidden_states.float(), gate.weight.float()),
        )

        expected = (
            torch.ones((1, 1), dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.int32),
        )
        with patch.object(
            torch.ops.sgl_kernel,
            "biased_grouped_topk_cpu",
            create=True,
            return_value=expected,
        ) as native_topk:
            actual = biased_grouped_topk_cpu(
                hidden_states,
                logits,
                torch.zeros(2, dtype=torch.float32),
                topk=1,
                renormalize=False,
                num_expert_group=1,
                topk_group=1,
            )

        self.assertIs(actual, expected)
        native_topk.assert_called_once()

    def test_cpu_fp32_weight_accepts_bf16_activations(self):
        gate = self._gate(torch.float32)
        hidden_states = torch.tensor([[1.001, 1.002]], dtype=torch.bfloat16)

        with patch.object(ernie4, "_is_cpu", True):
            logits = gate(hidden_states)

        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(
            logits,
            torch.nn.functional.linear(hidden_states.float(), gate.weight),
        )


if __name__ == "__main__":
    unittest.main()
