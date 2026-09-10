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


def _build(batch: int, n_heads: int, rope_dim: int):
    torch.manual_seed(0)
    x = torch.randn(batch, n_heads, rope_dim, device="cuda", dtype=torch.float32)
    freqs = precompute_freqs_cis(rope_dim, batch, 0, 10000.0, 1.0, 32, 1).to("cuda")
    return x, freqs


def _reference(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_real = x[..., 0::2].float()
    x_imag = x[..., 1::2].float()
    cos = freqs.real[:, None, :]
    sin = freqs.imag[:, None, :]
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    return torch.stack((out_real, out_imag), dim=-1).flatten(-2).to(x.dtype)


def _apply(x, freqs, batched: bool):
    previous = getattr(
        sys.modules[apply_rotary_emb_triton.__module__], "_USE_BATCHED_ROPE"
    )
    set_batched_rope(batched)
    try:
        return apply_rotary_emb_triton(x.clone(), freqs)
    finally:
        set_batched_rope(previous)


@pytest.mark.parametrize("rope_dim", [6, 64, 96, 128, 192])
@pytest.mark.parametrize("batch,n_heads", [(4, 8), (2, 4)])
class TestApplyRotaryEmbFlat:
    def test_matches_per_token_kernel(self, rope_dim: int, batch: int, n_heads: int):
        """The flat kernel must agree with the per-token kernel it replaces."""
        x, freqs = _build(batch, n_heads, rope_dim)
        reference = _apply(x, freqs, batched=False)
        result = _apply(x, freqs, batched=True)
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)

    def test_leaves_padding_columns_alone(
        self, rope_dim: int, batch: int, n_heads: int
    ):
        """The rotation must not reach past rope_dim into the next head's data."""
        x, freqs = _build(batch, n_heads, rope_dim)
        padded = torch.full(
            (batch, n_heads, 2 * rope_dim), -12345.0, device="cuda", dtype=torch.float32
        )
        padded[:, :, :rope_dim] = x
        view = padded[:, :, :rope_dim]

        set_batched_rope(True)
        try:
            apply_rotary_emb_triton(view, freqs)
        finally:
            set_batched_rope(False)

        untouched = padded[:, :, rope_dim:]
        assert torch.equal(
            untouched, torch.full_like(untouched, -12345.0)
        ), (
            f"{int((untouched != -12345.0).sum())} sentinel elements "
            "past rope_dim were overwritten"
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_independent_reference_and_preserves_suffix(
        self, rope_dim: int, batch: int, n_heads: int, dtype: torch.dtype
    ):
        """Only the first rope_dim columns of each head may change."""
        torch.manual_seed(0)
        x = torch.randn(batch, n_heads, rope_dim, device="cuda", dtype=dtype)
        freqs = precompute_freqs_cis(
            rope_dim, batch, 0, 10000.0, 1.0, 32, 1
        ).to("cuda")
        padded = torch.full(
            (batch, n_heads, 2 * rope_dim), -12345.0, device="cuda", dtype=dtype
        )
        padded[:, :, :rope_dim] = x
        view = padded[:, :, :rope_dim]

        set_batched_rope(True)
        try:
            apply_rotary_emb_triton(view, freqs)
        finally:
            set_batched_rope(False)

        reference = _reference(x, freqs)
        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(view, reference, rtol=tolerance, atol=tolerance)
        untouched = padded[:, :, rope_dim:]
        assert torch.equal(untouched, torch.full_like(untouched, -12345.0))

    def test_packed_view_matches_unpacked_and_preserves_sentinel(
        self, rope_dim: int, batch: int, n_heads: int
    ):
        """A stride-two packed view must match its unpacked contiguous clone."""
        x, freqs = _build(batch, n_heads, rope_dim)
        packed = torch.full(
            (batch, n_heads, 2 * rope_dim), -12345.0, device="cuda", dtype=torch.float32
        )
        packed[:, :, 0::2] = x
        view = packed[:, :, 0::2]
        unpacked = view.clone()

        set_batched_rope(True)
        try:
            apply_rotary_emb_triton(view, freqs)
        finally:
            set_batched_rope(False)

        reference = _apply(unpacked, freqs, batched=False)
        torch.testing.assert_close(view, reference, rtol=1e-5, atol=1e-5)
        sentinel = packed[:, :, 1::2]
        assert torch.equal(sentinel, torch.full_like(sentinel, -12345.0))

    def test_noncontiguous_head_stride_matches_unpacked(
        self, rope_dim: int, batch: int, n_heads: int
    ):
        """A strided head view must match its unpacked contiguous clone."""
        x, freqs = _build(batch, n_heads, rope_dim)
        packed = torch.empty(
            (batch, n_heads, 2, rope_dim), device="cuda", dtype=torch.float32
        )
        packed[:, :, 0] = x
        packed[:, :, 1] = -12345.0
        view = packed[:, :, 0, :]
        unpacked = view.clone()

        set_batched_rope(True)
        try:
            apply_rotary_emb_triton(view, freqs)
        finally:
            set_batched_rope(False)

        reference = _apply(unpacked, freqs, batched=False)
        torch.testing.assert_close(view, reference, rtol=1e-5, atol=1e-5)
        sentinel = packed[:, :, 1, :]
        assert torch.equal(sentinel, torch.full_like(sentinel, -12345.0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
