import unittest
from unittest import mock

import torch

from sglang.srt.models import deepseek_v2 as deepseek_v2_module
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-a", runner_config="1-gpu-small")


def make_moe(*, tp_size=8, moe_ep_size=8, shared_expert_tp1=True):
    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    moe.tp_size = tp_size
    moe.moe_ep_size = moe_ep_size
    moe._shared_expert_tp1 = shared_expert_tp1
    return moe


class TestDeepseekV2SharedExpertTp1(unittest.TestCase):
    def run_helper(self, moe, routed, shared, *, skip):
        with (
            mock.patch.object(
                deepseek_v2_module,
                "should_skip_post_experts_all_reduce",
                return_value=skip,
            ),
            mock.patch.object(
                deepseek_v2_module,
                "tensor_model_parallel_all_reduce",
                side_effect=lambda value: value,
            ) as all_reduce,
        ):
            output = moe._reduce_and_add_tp1_shared_output(routed.clone(), shared)
        return output, all_reduce

    def test_skipped_reduction_prescales_replicated_shared_output(self):
        routed = torch.tensor([[1.0, -2.0]])
        shared = torch.tensor([[0.5, 3.0]])
        moe = make_moe()

        output, all_reduce = self.run_helper(moe, routed, shared, skip=True)

        all_reduce.assert_not_called()
        torch.testing.assert_close(output, routed + shared / 8)

    def test_downstream_sum_has_one_shared_contribution(self):
        ep_size = 8
        shared = torch.tensor([[0.25, -1.5]])
        partials = [torch.tensor([[float(rank), rank / 2]]) for rank in range(ep_size)]
        moe = make_moe(tp_size=ep_size, moe_ep_size=ep_size)

        outputs = [
            self.run_helper(moe, partial, shared, skip=True)[0] for partial in partials
        ]

        torch.testing.assert_close(
            torch.stack(outputs).sum(0), torch.stack(partials).sum(0) + shared
        )

    def test_executed_all_reduce_keeps_full_shared_output(self):
        routed = torch.tensor([[1.0, -2.0]])
        shared = torch.tensor([[0.5, 3.0]])
        moe = make_moe()

        output, all_reduce = self.run_helper(moe, routed, shared, skip=False)

        all_reduce.assert_called_once()
        torch.testing.assert_close(output, routed + shared)

    def test_nonreplicated_and_single_rank_boundaries(self):
        routed = torch.tensor([[1.0, -2.0]])
        shared = torch.tensor([[0.5, 3.0]])

        nonreplicated = make_moe(shared_expert_tp1=False)
        output, _ = self.run_helper(nonreplicated, routed, shared, skip=True)
        torch.testing.assert_close(output, routed)

        single_rank = make_moe(tp_size=1, moe_ep_size=1)
        output, all_reduce = self.run_helper(single_rank, routed, shared, skip=True)
        all_reduce.assert_not_called()
        torch.testing.assert_close(output, routed + shared)

    def test_gpu_downstream_sum_matches_independent_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("requires GPU")
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(31475)
        ep_size = 8
        partials = torch.randn(8, 32, device=device, generator=generator)
        shared = torch.randn(32, device=device, generator=generator)
        moe = make_moe(tp_size=ep_size, moe_ep_size=ep_size)

        actual = torch.stack(
            [
                self.run_helper(moe, partial, shared, skip=True)[0]
                for partial in partials
            ]
        ).sum(0)
        reference = partials.double().cpu().sum(0) + shared.double().cpu()

        torch.testing.assert_close(
            actual.double().cpu(), reference, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
