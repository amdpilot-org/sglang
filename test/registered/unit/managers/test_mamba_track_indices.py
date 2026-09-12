from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    ("next_track_indices", "track_positions", "expected_positions", "expected_slots"),
    [
        ([None], None, [0], [11]),
        ([0, 1, None], None, [0, 1, 0], [11, 22, 31]),
        ([None, None], [1, 0], [1, 0], [12, 21]),
    ],
)
def test_set_mamba_track_indices_handles_uninitialized_positions(
    next_track_indices, track_positions, expected_positions, expected_slots
):
    """An uninitialized request in TARGET_VERIFY must not reach torch.tensor as None."""
    device = torch.device("cuda")
    mapping = torch.tensor(
        [[11, 12], [21, 22], [31, 32]], dtype=torch.int64, device=device
    )
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
