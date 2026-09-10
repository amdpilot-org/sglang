"""TARGET_VERIFY mamba track metadata uses the intended state destinations."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.speculative.spec_utils import (  # noqa: E402
    prepare_mamba_track_for_verify,
)

register_cuda_ci(est_time=1, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _make_batch(next_track_indices):
    reqs = [
        SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=index))
        for index in next_track_indices
    ]
    batch = ScheduleBatch(reqs=reqs)
    batch.req_pool_indices = torch.tensor([7, 8], dtype=torch.int64, device="cuda")
    batch.req_to_token_pool = SimpleNamespace(
        req_index_to_mamba_ping_pong_track_buffer_mapping=torch.arange(
            32, dtype=torch.int64, device="cuda"
        ).view(16, 2)
    )
    batch.mamba_track_mask = torch.ones(2, dtype=torch.bool, device="cuda")
    batch.mamba_track_seqlens = torch.ones(2, dtype=torch.int64, device="cuda")
    return batch


class TestMambaTargetVerifyTrackIndices(CustomTestCase):
    def test_absent_next_track_index_defaults_to_first_slot(self):
        batch = _make_batch([None, None])

        with get_context().override_server_args(
            mamba_radix_cache_strategy="extra_buffer"
        ):
            prepare_mamba_track_for_verify(batch)

        self.assertEqual(batch.mamba_track_buffer_indices, [0, 0])
        self.assertEqual(batch.mamba_track_indices.tolist(), [14, 16])
        self.assertIsNone(batch.mamba_track_mask)
        self.assertIsNone(batch.mamba_track_seqlens)

    def test_lazy_plan_selects_planned_state_destination(self):
        batch = _make_batch([0, 1])
        batch.mamba_lazy_spec_track_positions_cpu = [1, 0]

        with get_context().override_server_args(
            mamba_radix_cache_strategy="extra_buffer_lazy"
        ):
            prepare_mamba_track_for_verify(batch)

        self.assertEqual(batch.mamba_track_buffer_indices, [1, 0])
        self.assertEqual(batch.mamba_track_indices.tolist(), [15, 16])
        self.assertIsNone(batch.mamba_track_mask)
        self.assertIsNone(batch.mamba_track_seqlens)


if __name__ == "__main__":
    unittest.main()
