"""Correctness test for shared-prefix/tail attention decomposition.

The test compares a decode-style attention output computed over a shared prefix
and per-request tail with a full-attention reference. The two partial results
are combined with a stable pure-Torch LSE merge, which is supported on both
NVIDIA GPUs and AMD ROCm GPUs.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci


register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    "batch_size,prefix_len,tail_len,num_heads,head_dim,dtype",
    [
        (2, 128, 16, 4, 64, torch.float16),
        (4, 256, 32, 8, 64, torch.bfloat16),
        (8, 512, 64, 8, 64, torch.float16),
    ],
)
def test_shared_prefix_lse_decomposition_matches_full_attention(
    batch_size: int,
    prefix_len: int,
    tail_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
):
    torch.manual_seed(1715)
    device = torch.device("cuda")
    scale = head_dim**-0.5

    query = torch.randn(
        batch_size, num_heads, head_dim, device=device, dtype=dtype
    )
    prefix_key = torch.randn(
        prefix_len, num_heads, head_dim, device=device, dtype=dtype
    )
    prefix_value = torch.randn(
        prefix_len, num_heads, head_dim, device=device, dtype=dtype
    )
    tail_key = torch.randn(
        batch_size, tail_len, num_heads, head_dim, device=device, dtype=dtype
    )
    tail_value = torch.randn(
        batch_size, tail_len, num_heads, head_dim, device=device, dtype=dtype
    )
    full_value = torch.cat(
        [prefix_value.unsqueeze(0).expand(batch_size, -1, -1, -1), tail_value],
        dim=1,
    )

    prefix_scores = torch.einsum(
        "bhd,khd->bhk", query.float(), prefix_key.float()
    ) * scale
    tail_scores = torch.einsum(
        "bhd,bthd->bht", query.float(), tail_key.float()
    ) * scale
    full_scores = torch.cat([prefix_scores, tail_scores], dim=2)

    prefix_weights = torch.softmax(prefix_scores, dim=-1)
    tail_weights = torch.softmax(tail_scores, dim=-1)
    full_weights = torch.softmax(full_scores, dim=-1)

    prefix_output = torch.einsum(
        "bhk,khd->bhd", prefix_weights, prefix_value.float()
    ).to(dtype)
    tail_output = torch.einsum(
        "bht,bthd->bhd", tail_weights, tail_value.float()
    ).to(dtype)
    full_output = torch.einsum(
        "bhk,bkhd->bhd", full_weights, full_value.float()
    ).to(dtype)

    prefix_lse = torch.logsumexp(prefix_scores, dim=-1)
    tail_lse = torch.logsumexp(tail_scores, dim=-1)
    full_lse = torch.logsumexp(full_scores, dim=-1)

    max_lse = torch.maximum(prefix_lse, tail_lse)
    prefix_weight = torch.exp(prefix_lse - max_lse)
    tail_weight = torch.exp(tail_lse - max_lse)
    total_weight = prefix_weight + tail_weight
    merged_output = (
        prefix_output.float() * (prefix_weight / total_weight).unsqueeze(-1)
        + tail_output.float() * (tail_weight / total_weight).unsqueeze(-1)
    ).to(dtype)
    merged_lse = max_lse + torch.log(total_weight)

    output_atol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    output_rtol = 2e-2 if dtype == torch.bfloat16 else 2e-2
    torch.testing.assert_close(
        merged_output.float(),
        full_output.float(),
        atol=output_atol,
        rtol=output_rtol,
    )
    torch.testing.assert_close(
        merged_lse.float(),
        full_lse.float(),
        atol=1e-4,
        rtol=1e-4,
    )
