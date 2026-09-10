"""GPU tests for model-specific MoE router precision contracts."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.bailing_moe import BailingMoEGate
from sglang.srt.models.bailing_moe_linear import BailingMoEGate as BailingLinearMoEGate
from sglang.srt.models.llada2 import LLaDA2MoeGate
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="jit-kernel-unit", runner_config="amd")


class TestRouterLogitsNumerics(CustomTestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_model_gates_share_fp32_reference_and_preserve_near_ties(self):
        gate_types = (BailingMoEGate, BailingLinearMoEGate, LLaDA2MoeGate)
        config = SimpleNamespace(
            num_experts=8,
            hidden_size=16,
            moe_router_enable_expert_bias=False,
        )
        hidden_states = torch.ones(4, 16, dtype=torch.bfloat16, device="cuda")
        weight = torch.zeros(8, 16, dtype=torch.float32, device="cuda")
        weight[0] = (1.0 + 1e-5) / 16
        weight[1] = 1.0 / 16
        weight[2] = 0.5 / 16
        weight[3] = 0.25 / 16
        reference = hidden_states.float() @ weight.T
        expected_topk = reference.topk(2).indices
        bf16_topk = reference.bfloat16().topk(2).indices

        self.assertEqual(reference.dtype, torch.float32)
        self.assertFalse(torch.equal(expected_topk, bf16_topk))

        for gate_type in gate_types:
            with self.subTest(gate_type=gate_type.__name__):
                gate = gate_type(config=config, params_dtype=torch.float32).to("cuda")
                with torch.no_grad():
                    gate.weight.copy_(weight)
                    logits = gate(hidden_states)

                self.assertEqual(logits.dtype, torch.float32)
                torch.testing.assert_close(logits, reference, atol=1e-6, rtol=1e-6)
                self.assertTrue(torch.equal(logits.topk(2).indices, expected_topk))


if __name__ == "__main__":
    unittest.main()
