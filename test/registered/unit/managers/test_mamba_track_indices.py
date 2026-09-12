import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_batch(track_positions, *, device="cpu"):
    mapping = torch.tensor(
        [[10, 11], [20, 21], [30, 31]], dtype=torch.int64, device=device
    )
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
        ),
        req_pool_indices=torch.arange(len(track_positions), device=device),
        reqs=[
            SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=position))
            for position in track_positions
        ],
    )


class TestMambaTrackIndices(unittest.TestCase):
    def test_unallocated_positions_use_first_ping_pong_slot(self):
        batch = _make_batch([None, None, None])

        set_mamba_track_indices_from_reqs(batch)

        self.assertEqual(batch.mamba_track_buffer_indices, [0, 0, 0])
        self.assertEqual(batch.mamba_track_indices.tolist(), [10, 20, 30])

    def test_unallocated_position_does_not_replace_valid_positions(self):
        batch = _make_batch([None, 1, 0])

        set_mamba_track_indices_from_reqs(batch)

        self.assertEqual(batch.mamba_track_buffer_indices, [0, 1, 0])
        self.assertEqual(batch.mamba_track_indices.tolist(), [10, 21, 30])

    def test_explicit_track_positions_remain_authoritative(self):
        batch = _make_batch([None, None, None])

        set_mamba_track_indices_from_reqs(batch, track_positions=[1, 0, 1])

        self.assertEqual(batch.mamba_track_buffer_indices, [1, 0, 1])
        self.assertEqual(batch.mamba_track_indices.tolist(), [11, 20, 31])


if __name__ == "__main__":
    unittest.main()
