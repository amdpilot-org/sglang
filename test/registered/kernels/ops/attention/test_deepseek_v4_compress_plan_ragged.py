"""Regression tests for ragged GPU prefill-plan min/max reduction.

The GPU planner must not mistake a non-uniform extend batch for the uniform MTP
fast path.  Such a misclassification derives ``ragged_id`` as ``batch * E + j``
and can address beyond the compact ragged input.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import make_paged_context, to_seq_extend
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _valid_rows(plan: torch.Tensor, words_per_row: int) -> list[tuple[int, ...]]:
    words = plan.cpu().view(torch.uint32).view(-1, words_per_row)
    return sorted(
        tuple(int(value) for value in row)
        for row in words
        if int(row[0]) != 0xFFFFFFFF
    )


def _assert_warp_extrema_synchronized(test: unittest.TestCase, kernel: str) -> None:
    """Accept no initialization, or initialization ordered before reduction."""
    reduction = kernel.split("// === Stage B:", 1)[1].split(
        "const auto num_q", 1
    )[0]
    test.assertEqual(reduction.count("warp_max["), 2)
    test.assertEqual(reduction.count("warp_min["), 2)

    before_reduction = kernel.split("// === Stage B:", 1)[0]
    init_positions = [
        before_reduction.find("warp_max[tx] ="),
        before_reduction.find("warp_min[tx] ="),
    ]
    if any(position >= 0 for position in init_positions):
        test.assertTrue(all(position >= 0 for position in init_positions))
        barrier = before_reduction.rfind("__syncthreads();")
        test.assertGreater(barrier, max(init_positions))


class TestCompressPlanRaggedReduction(CustomTestCase):
    def test_warp_extrema_initialization_is_synchronized(self):
        """Reject only unsynchronized scratch initialization.

        The current implementation needs no initialization because every warp
        writes its own slot.  Initialization is also correct when a block barrier
        orders it before those per-warp writes, as in the initial upstream fix.
        """
        source = (
            Path(__file__).parents[5]
            / "python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh"
        ).read_text()
        kernel = source.split("void plan_compress_prefill_kernel0", 1)[1].split(
            "__global__ void plan_compress_prefill_kernel_1", 1
        )[0]
        _assert_warp_extrema_synchronized(self, kernel)

    def test_barrier_based_initialization_boundary(self):
        source = (
            Path(__file__).parents[5]
            / "python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh"
        ).read_text()
        kernel = source.split("void plan_compress_prefill_kernel0", 1)[1].split(
            "__global__ void plan_compress_prefill_kernel_1", 1
        )[0]
        stage_b = "// === Stage B:"
        initialization = """if (tx < kNumWarps) {
    warp_max[tx] = 0;
    warp_min[tx] = 0xFFFFFFFFu;
  }
  """

        safe_kernel = kernel.replace(
            stage_b, initialization + "__syncthreads();\n  " + stage_b
        )
        _assert_warp_extrema_synchronized(self, safe_kernel)

        racy_kernel = kernel.replace(stage_b, initialization + stage_b)
        with self.assertRaises(AssertionError):
            _assert_warp_extrema_synchronized(self, racy_kernel)

    def _assert_gpu_matches_cpu(self, extend_lens: list[int]) -> None:
        bs = len(extend_lens)
        ctx = make_paged_context(
            bs=bs,
            compress_ratio=128,
            ring_size=256,
            num_reqs_capacity=bs,
        )
        # Keep prefixes distinct enough to exercise final location translation while
        # staying within the fixture's req_to_token table.
        seq_extend = [(512 + (i % 7) + extend, extend) for i, extend in enumerate(extend_lens)]
        seq_lens, extends, num_q = to_seq_extend(seq_extend)

        cpu_plan = ctx.make_prefill_plan(seq_lens, extends, num_q)
        gpu_plan = ctx.make_prefill_plan(
            seq_lens.to(ctx.req_to_token.device),
            extends.to(ctx.req_to_token.device),
            num_q,
        )
        torch.cuda.synchronize()

        cpu_c = _valid_rows(cpu_plan.plan_c, 4)
        gpu_c = _valid_rows(gpu_plan.plan_c, 4)
        cpu_w = _valid_rows(cpu_plan.plan_w, 2)
        gpu_w = _valid_rows(gpu_plan.plan_w, 2)
        self.assertEqual(gpu_c, cpu_c)
        self.assertEqual(gpu_w, cpu_w)

        ragged_ids = [row[0] for row in gpu_w]
        self.assertTrue(all(ragged_id < num_q for ragged_id in ragged_ids))
        self.assertEqual(len(ragged_ids), len(set(ragged_ids)))

    def test_compact_capture_report_pattern(self):
        # The issue's bs=96 compact capture shape: 72 requests at E=4 followed
        # by 24 requests at E=3.  The transition crosses multiple warps.
        for _ in range(100):
            self._assert_gpu_matches_cpu([4] * 72 + [3] * 24)

    def test_nonuniform_at_warp_boundaries(self):
        for boundary in (31, 32, 33, 63, 64, 65):
            with self.subTest(boundary=boundary):
                self._assert_gpu_matches_cpu([4] * boundary + [3] * (96 - boundary))

    def test_uniform_fast_path_control(self):
        for extend in (1, 4, 32):
            with self.subTest(extend=extend):
                self._assert_gpu_matches_cpu([extend] * 96)


if __name__ == "__main__":
    unittest.main()
