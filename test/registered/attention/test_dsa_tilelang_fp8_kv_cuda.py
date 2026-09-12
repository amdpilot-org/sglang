"""Numerical regression for the CUDA raw-FP8 TileLang DSA path."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

tilelang_kernel = pytest.importorskip(
    "sglang.kernels.ops.attention.dsa.tilelang_kernel"
)

requires_fp8_cuda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() < (8, 9),
    reason="needs CUDA SM89+ for FP8 tensor-core MMA",
)

S, H, DV, TOPK, POOL = 4, 32, 512, 2112, 32768
SM_SCALE = DV**-0.5


def _relative_error(actual, expected):
    return (
        (actual.float() - expected.float()).abs().max()
        / expected.float().abs().max()
    ).item()


def _make_inputs(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        S, H, DV, device="cuda", dtype=torch.float32, generator=generator
    ) * 0.5
    kv = torch.randn(
        POOL, 1, DV, device="cuda", dtype=torch.float32, generator=generator
    ) * 0.5
    indices = torch.randint(
        1, POOL, (S, 1, TOPK), device="cuda", generator=generator
    ).to(torch.int32)
    indices[:, :, -37:] = -1
    return q.to(torch.bfloat16), kv.to(torch.bfloat16), indices


def _reference_attention(q, kv, indices):
    output = torch.empty(S, H, DV, device="cuda", dtype=torch.float32)
    kv = kv.squeeze(1)
    for batch_index in range(S):
        token_ids = indices[batch_index, 0].long()
        valid = token_ids >= 0
        selected_kv = kv[token_ids.clamp(min=0)]
        logits = (q[batch_index] @ selected_kv.T) * SM_SCALE
        logits[:, ~valid] = float("-inf")
        output[batch_index] = torch.softmax(logits, dim=-1) @ selected_kv
    return output


@requires_fp8_cuda
def test_fp8_one_hot_exact():
    q, kv, _ = _make_inputs(0)
    kv_fp8 = kv.to(torch.float8_e4m3fn)
    indices = torch.full((S, 1, TOPK), -1, device="cuda", dtype=torch.int32)
    selected = torch.randint(1, POOL, (S,), device="cuda")
    indices[:, 0, 0] = selected.to(torch.int32)

    output = tilelang_kernel.tilelang_sparse_fwd(
        q, kv_fp8, indices, SM_SCALE, d_v=DV
    )
    torch.cuda.synchronize()
    expected = kv_fp8.float().squeeze(1)[selected.long()].unsqueeze(1)
    expected = expected.expand(S, H, DV)
    assert _relative_error(output.reshape(S, H, DV), expected) < 1e-3


@requires_fp8_cuda
def test_fp8_spread_and_scrambled_indices_negative_control():
    q, kv, indices = _make_inputs(1)
    kv_fp8 = kv.to(torch.float8_e4m3fn)
    output = tilelang_kernel.tilelang_sparse_fwd(
        q, kv_fp8, indices, SM_SCALE, d_v=DV
    )
    torch.cuda.synchronize()
    reference = _reference_attention(
        q.to(torch.float8_e4m3fn).float(), kv_fp8.float(), indices
    )
    assert _relative_error(output.reshape(S, H, DV), reference) < 0.04

    scrambled = indices.clone()
    scrambled[:, :, : TOPK // 2] = indices[:, :, TOPK // 2 : TOPK].flip(-1)[
        :, :, : TOPK // 2
    ]
    wrong = tilelang_kernel.tilelang_sparse_fwd(
        q, kv_fp8, scrambled, SM_SCALE, d_v=DV
    )
    torch.cuda.synchronize()
    assert _relative_error(wrong.reshape(S, H, DV), reference) > 0.04
