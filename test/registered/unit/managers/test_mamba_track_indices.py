from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    ("next_track_indices", "track_positions", "expected_positions", "expected_slots"),
    [
        ([0, 1], None, [0, 1], [11, 22]),
        ([None, 1], None, [0, 1], [-1, 22]),
        ([None, None], [1, 0], [1, 0], [-1, -1]),
    ],
)
def test_set_mamba_track_indices_skips_freed_rows(
    next_track_indices, track_positions, expected_positions, expected_slots
):
    """A verify row whose request state was freed must not reuse its stale slot."""
    device = torch.device("cuda")
    mapping = torch.tensor([[11, 12], [21, 22]], dtype=torch.int64, device=device)
    batch_size = len(next_track_indices)
    batch = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
        ),
        req_pool_indices=torch.arange(batch_size, device=device),
        reqs=[
            SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=position))
            for position in next_track_indices
        ],
    )

    set_mamba_track_indices_from_reqs(batch, track_positions)
    torch.cuda.synchronize()

    assert batch.mamba_track_buffer_indices == expected_positions
    assert batch.mamba_track_indices.cpu().tolist() == expected_slots
