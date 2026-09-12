"""Regression coverage for ragged DSV4 GPU prefill planning.

The c128 planner has a uniform-MTP fast path.  A former shared-memory race in
its min/max reduction could misclassify ragged ``extend_lens`` as uniform and
emit an out-of-bounds ``ragged_id``.  Keep these inputs on the GPU so this test
exercises ``plan_compress_prefill_kernel0``, not the CPU planning fallback.
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import make_paged_context
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _valid_write_ragged_ids(plan_w: torch.Tensor) -> torch.Tensor:
    # WritePlan is two uint32 words: ragged_id followed by write_loc.  Use int64
    # for masking because ROCm does not implement boolean indexing for uint32.
    ragged_ids = plan_w.view(torch.int32).view(-1, 2)[:, 0].to(torch.int64)
    ragged_ids &= 0xFFFFFFFF
    return ragged_ids[ragged_ids != 0xFFFFFFFF]


class TestCompressPlanRagged(CustomTestCase):
    def _assert_ragged_plan(self, extend_lens: list[int], iterations: int = 100):
        ctx = make_paged_context(
            bs=len(extend_lens),
            compress_ratio=128,
            num_reqs_capacity=len(extend_lens),
        )
        extend = torch.tensor(extend_lens, dtype=torch.int64, device="cuda")
        seq = extend.clone()
        num_q = sum(extend_lens)
        expected = torch.arange(num_q, dtype=torch.int64, device="cuda")

        for _ in range(iterations):
            plan = ctx.make_prefill_plan(seq, extend, num_q)
            actual = _valid_write_ragged_ids(plan.plan_w).sort().values
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reported_ragged_shapes(self):
        self._assert_ragged_plan([4] * 72 + [3] * 24)
        self._assert_ragged_plan([3] * 104 + [2] * 24)

    def test_uniform_control(self):
        self._assert_ragged_plan([4] * 96)

    def test_warp_and_block_boundaries(self):
        # A partial warp, all 32 warps in the fixed 1024-thread launch, and a
        # zero-length request independently cover reduction boundary behavior.
        self._assert_ragged_plan([4] * 30 + [3])
        self._assert_ragged_plan([4] * 1023 + [3])
        self._assert_ragged_plan([4] * 31 + [0])


if __name__ == "__main__":
    unittest.main()
