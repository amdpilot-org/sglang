"""Execution-representation checks for the ROCm router GEMM."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

try:
    from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm
except ImportError:
    aiter_dsv3_router_gemm = None

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

NUM_EXPERTS = 256
HIDDEN_DIM = 7168
ROW_PADDING = 8


def _guarded_row_sliced_hidden_states(num_tokens):
    guard = torch.full((16,), 123.25, device="cuda", dtype=torch.bfloat16)
    storage = torch.empty(
        guard.numel()
        + num_tokens * (HIDDEN_DIM + ROW_PADDING)
        + guard.numel(),
        device="cuda",
        dtype=torch.bfloat16,
    )
    storage[: guard.numel()] = guard
    storage[-guard.numel() :] = guard
    base = storage[guard.numel() : -guard.numel()].view(
        num_tokens, HIDDEN_DIM + ROW_PADDING
    )
    base.copy_(torch.randn_like(base))
    return guard, storage, base[:, :HIDDEN_DIM]


@unittest.skipUnless(
    aiter_dsv3_router_gemm is not None and torch.cuda.is_available(),
    "ROCm AITER router GEMM and one GPU are required",
)
class TestROCmRouterGEMMExecutionRepresentations(unittest.TestCase):
    def test_row_sliced_hidden_states_match_reference(self):
        weight = torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16
        )
        for num_tokens in (1, 8, 16):
            with self.subTest(num_tokens=num_tokens):
                guard, storage, hidden_states = _guarded_row_sliced_hidden_states(
                    num_tokens
                )
                reference = (
                    hidden_states.double().cpu() @ weight.double().cpu().T
                ).to(torch.bfloat16)
                output = aiter_dsv3_router_gemm(hidden_states, weight)
                torch.cuda.synchronize()

                self.assertEqual(output.dtype, hidden_states.dtype)
                torch.testing.assert_close(
                    output.float(),
                    reference.float().cuda(),
                    atol=1.0,
                    rtol=1e-2,
                )
                self.assertTrue(
                    torch.equal(storage[: guard.numel()], guard)
                    and torch.equal(storage[-guard.numel() :], guard)
                )

    def test_inner_strided_hidden_states_fail_clearly(self):
        hidden_states = torch.randn(
            1, HIDDEN_DIM * 2, device="cuda", dtype=torch.bfloat16
        )[:, ::2]
        weight = torch.randn(
            NUM_EXPERTS, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16
        )
        with self.assertRaisesRegex(
            RuntimeError, "hidden_states to have a unit inner stride"
        ):
            aiter_dsv3_router_gemm(hidden_states, weight)

    def test_non_contiguous_weight_fails_clearly(self):
        hidden_states = torch.randn(
            1, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            NUM_EXPERTS,
            HIDDEN_DIM + ROW_PADDING,
            device="cuda",
            dtype=torch.bfloat16,
        )[:, :HIDDEN_DIM]
        with self.assertRaisesRegex(RuntimeError, "contiguous weight"):
            aiter_dsv3_router_gemm(hidden_states, weight)


if __name__ == "__main__":
    unittest.main()
