from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU-backed mamba buffer"
)


def _batch(track_positions):
    buffers = torch.tensor(
        [[101, 102], [201, 202], [301, 302]],
        dtype=torch.int64,
        device="cuda",
    )
    reqs = [
        SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=position))
        for position in track_positions
    ]
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=buffers
        ),
        req_pool_indices=torch.arange(len(reqs), device="cuda"),
        reqs=reqs,
    )


def test_fresh_request_defaults_to_first_mamba_track_slot():
    batch = _batch([None])

    set_mamba_track_indices_from_reqs(batch)

    assert batch.mamba_track_buffer_indices == [0]
    assert batch.mamba_track_indices.cpu().tolist() == [101]
    # Building verify metadata must not claim that request allocation occurred.
    assert batch.reqs[0].kv.mamba_next_track_idx is None


def test_mamba_track_indices_keep_initialized_slots_in_mixed_batch():
    batch = _batch([None, 1, 0])

    set_mamba_track_indices_from_reqs(batch)

    assert batch.mamba_track_buffer_indices == [0, 1, 0]
    assert batch.mamba_track_indices.cpu().tolist() == [101, 202, 301]


def test_lazy_spec_track_plan_overrides_request_slots():
    batch = _batch([0, 1, None])

    set_mamba_track_indices_from_reqs(batch, track_positions=[1, 0, 1])

    assert batch.mamba_track_buffer_indices == [1, 0, 1]
    assert batch.mamba_track_indices.cpu().tolist() == [102, 201, 302]
