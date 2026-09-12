import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sglang.srt.models.phi4mm_utils import adaptive_enc_mask


def _reference_adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    """The interval/nonzero implementation used before the optimization."""
    chunk_start_idx = torch.Tensor(chunk_start_idx).long()
    start_pad = F.pad(chunk_start_idx, (1, 0))
    end_pad = F.pad(chunk_start_idx, (0, 1), value=x_len)
    seq_range = torch.arange(0, x_len).unsqueeze(-1)
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]

    seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    mask_left = seq_range_expand >= start_pad[idx_left].unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    mask_right = seq_range_expand < end_pad[idx_right].unsqueeze(-1)
    return mask_left & mask_right


@pytest.mark.parametrize(
    ("x_len", "chunk_start_idx"),
    [
        (0, np.array([], dtype=np.int64)),  # empty
        (1, np.array([0])),  # lower and upper boundary coincide
        (18, np.array([0])),  # one sparse chunk
        (19, np.array([0, 18])),  # position immediately after a boundary
        (32, np.arange(32)),  # dense: every position starts a chunk
        (257, np.arange(0, 257, 18)),  # normal streaming layout and tail
    ],
)
@pytest.mark.parametrize(("left_window", "right_window"), [(0, 0), (6, 0), (1, 1)])
def test_adaptive_enc_mask_matches_interval_reference(
    x_len, chunk_start_idx, left_window, right_window
):
    actual = adaptive_enc_mask(
        x_len, chunk_start_idx, left_window=left_window, right_window=right_window
    )
    expected = _reference_adaptive_enc_mask(
        x_len, chunk_start_idx, left_window=left_window, right_window=right_window
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (x_len, x_len)
    assert actual.dtype == torch.bool
    assert actual.device.type == "cpu"


def test_bucketize_preserves_nonzero_index_order_and_metadata():
    x_len = 73
    chunk_start_idx = torch.tensor([0, 1, 18, 36, 72], dtype=torch.long)
    start_pad = F.pad(chunk_start_idx, (1, 0))
    end_pad = F.pad(chunk_start_idx, (0, 1), value=x_len)
    seq_range = torch.arange(x_len)

    expected = (
        (seq_range.unsqueeze(-1) < end_pad) & (seq_range.unsqueeze(-1) >= start_pad)
    ).nonzero()[:, 1]
    actual = torch.bucketize(seq_range, chunk_start_idx, right=True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == expected.shape == (x_len,)
    assert actual.dtype == expected.dtype == torch.int64
    assert actual.device == expected.device
