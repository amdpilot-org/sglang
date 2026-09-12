import pytest

from sglang.srt.layers.attention.dsa.trtllm_utils import (
    TRTLLM_MLA_MAX_BATCH_SIZE,
    validate_trtllm_mla_batch_size,
)


@pytest.mark.parametrize("batch_size", [1, 1024, 65534, 65535])
def test_validate_trtllm_mla_batch_size_accepts_valid_boundaries(batch_size):
    validate_trtllm_mla_batch_size(batch_size)


@pytest.mark.parametrize("batch_size", [65536, 65537, 131072])
def test_validate_trtllm_mla_batch_size_rejects_grid_dim_z_overflow(batch_size):
    with pytest.raises(
        ValueError,
        match=rf"got {batch_size} query rows.*limit of {TRTLLM_MLA_MAX_BATCH_SIZE}",
    ):
        validate_trtllm_mla_batch_size(batch_size)
