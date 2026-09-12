import sys

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import (
    apply_rotary_emb_triton,
    precompute_freqs_cis,
    set_batched_rope,
)
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _inputs(batch: int, n_heads: int, rope_dim: int):
    torch.manual_seed(0)
    x = torch.randn(batch, n_heads, rope_dim, device="cuda", dtype=torch.float32)
    freqs = precompute_freqs_cis(rope_dim, batch, 0, 10000.0, 1.0, 32, 1).to(
        "cuda"
    )
    return x, freqs


def _torch_reference(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = pairs * freqs[:, None, :]
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


@pytest.fixture(autouse=True)
def _restore_kernel_selection():
    set_batched_rope(False)
    yield
    set_batched_rope(False)


@pytest.mark.parametrize("rope_dim", [6, 64, 96, 128, 192])
@pytest.mark.parametrize("batch,n_heads", [(4, 8), (1, 1)])
def test_flat_matches_torch_reference(rope_dim: int, batch: int, n_heads: int):
    """Cover padded widths plus power-of-two and final-row boundary controls."""
    x, freqs = _inputs(batch, n_heads, rope_dim)
    expected = _torch_reference(x, freqs)

    set_batched_rope(True)
    actual = apply_rotary_emb_triton(x.clone(), freqs)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rope_dim", [6, 64, 96, 128, 192])
def test_flat_does_not_overwrite_columns_past_rope_dim(rope_dim: int):
    """Expose padded-column stores through a wider backing allocation."""
    batch, n_heads = 2, 4
    x, freqs = _inputs(batch, n_heads, rope_dim)
    sentinel = -12345.0
    backing = torch.full(
        (batch, n_heads, 2 * rope_dim),
        sentinel,
        device="cuda",
        dtype=torch.float32,
    )
    backing[:, :, :rope_dim] = x

    set_batched_rope(True)
    apply_rotary_emb_triton(backing[:, :, :rope_dim], freqs)

    tail = backing[:, :, rope_dim:]
    assert torch.equal(tail, torch.full_like(tail, sentinel)), (
        f"{int((tail != sentinel).sum())} elements past rope_dim were overwritten"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
