"""BF16 shared-plus-routed accumulation tests for the Triton MoE runner.

These tests use small synthetic tensors to compare the actual routing and
combination path with an independent PyTorch reference. They deliberately cover
zero and one routed expert in addition to the always-selected shared expert.
"""

import unittest

import torch

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import TritonKernelTopKOutput, routing
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestTritonSharedRoutedBF16Accumulation(CustomTestCase):
    TOKENS = 5
    HIDDEN_SIZE = 32
    INTERMEDIATE_SIZE = 16
    ROUTED_EXPERTS = 4
    SHARED_EXPERT_ID = 4

    def setUp(self):
        super().setUp()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _run_case(self, routed_experts: int):
        torch.manual_seed(1234 + routed_experts)
        device = "cuda"
        dtype = torch.bfloat16
        top_k = routed_experts + 1

        weight_scale = 0.1
        hidden_states = (
            torch.randn(self.TOKENS, self.HIDDEN_SIZE, device=device, dtype=dtype)
            * weight_scale
        )
        w13 = (
            torch.randn(
                self.ROUTED_EXPERTS + 1,
                2 * self.INTERMEDIATE_SIZE,
                self.HIDDEN_SIZE,
                device=device,
                dtype=dtype,
            )
            * weight_scale
        )
        w2 = (
            torch.randn(
                self.ROUTED_EXPERTS + 1,
                self.HIDDEN_SIZE,
                self.INTERMEDIATE_SIZE,
                device=device,
                dtype=dtype,
            )
            * weight_scale
        )
        w13_tri = w13.transpose(-2, -1).contiguous()
        w2_tri = w2.transpose(-2, -1).contiguous()

        router_logits = torch.randn(
            self.TOKENS, self.ROUTED_EXPERTS, device=device, dtype=dtype
        )
        shared_logits = torch.full(
            (self.TOKENS, 1), 4.0, device=device, dtype=dtype
        )
        router_logits = torch.cat([router_logits, shared_logits], dim=-1)

        (
            ragged_metadata,
            gather_indx,
            scatter_indx,
            gate_scal,
            actual_top_k,
        ) = routing(router_logits, top_k, sm_first=True)
        self.assertEqual(actual_top_k, top_k)

        actual_weights = torch.empty_like(gate_scal)
        actual_weights[scatter_indx.long()] = gate_scal
        actual_weights = actual_weights.view(self.TOKENS, top_k)

        expected_weights, expected_ids = torch.topk(
            torch.softmax(router_logits.float(), dim=-1), top_k, dim=-1
        )
        self.assertTrue(
            torch.equal(
                expected_ids[:, 0],
                torch.full_like(expected_ids[:, 0], self.SHARED_EXPERT_ID),
            )
        )
        torch.testing.assert_close(
            actual_weights,
            expected_weights.to(dtype),
            rtol=1e-2,
            atol=1e-3,
        )

        expected_expert_outputs = []
        for token_position, token_ids in enumerate(expected_ids):
            token_outputs = []
            for expert_id in token_ids:
                expert_output = SiluAndMul()(
                    hidden_states[token_position] @ w13[expert_id].transpose(0, 1)
                ) @ w2[expert_id].transpose(0, 1)
                token_outputs.append(expert_output)
            expected_expert_outputs.append(torch.stack(token_outputs))
        expected_expert_outputs = torch.stack(expected_expert_outputs)

        expected_weighted_outputs = expected_expert_outputs * expected_weights.to(
            dtype
        ).unsqueeze(-1)
        expected_shared_contribution = expected_weighted_outputs[:, 0]
        expected_combined = expected_weighted_outputs.sum(dim=1)

        topk_output = TritonKernelTopKOutput(
            ragged_metadata,
            gather_indx,
            scatter_indx,
            gate_scal,
            actual_top_k,
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )
        quant_info = TritonKernelsQuantInfo(
            w13_weight=w13_tri, w2_weight=w2_tri
        )

        combined_runner = MoeRunner(
            MoeRunnerBackend.TRITON_KERNELS, MoeRunnerConfig(inplace=False)
        )
        actual_combined = combined_runner.run(dispatch_output, quant_info).hidden_states

        torch.testing.assert_close(
            actual_combined,
            expected_combined,
            rtol=2e-2,
            atol=2e-2,
        )

    def test_zero_routed_expert(self):
        self._run_case(routed_experts=0)

    def test_one_routed_expert(self):
        self._run_case(routed_experts=1)


if __name__ == "__main__":
    unittest.main()
