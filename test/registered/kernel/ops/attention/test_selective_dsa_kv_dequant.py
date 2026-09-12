import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_selective,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _pack_cache(reference: torch.Tensor) -> torch.Tensor:
    """Independent packer for [512 fp8 | 4 fp32 scales | 64 bf16] cache rows."""
    assert reference.dtype == torch.bfloat16 and reference.shape[1] == 576
    nope = reference[:, :512].float().reshape(-1, 4, 128)
    scales = nope.abs().amax(dim=-1).clamp_min(1e-6) / 448.0
    quantized = (nope / scales.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(-1, 512)
    packed = torch.empty(
        (reference.shape[0], 656), dtype=torch.float8_e4m3fn, device=reference.device
    )
    packed[:, :512] = quantized
    packed[:, 512:528].view(torch.float32).copy_(scales)
    packed[:, 528:].view(torch.bfloat16).copy_(reference[:, 512:])
    return packed.unsqueeze(1)


def _reference_dequant(packed: torch.Tensor, physical: torch.Tensor) -> torch.Tensor:
    rows = packed.view(-1, 656).index_select(0, physical.long())
    nope = rows[:, :512].float().reshape(-1, 4, 128)
    scales = rows[:, 512:528].view(torch.float32)
    rope = rows[:, 528:].view(torch.bfloat16)
    return torch.cat(
        [(nope * scales.unsqueeze(-1)).reshape(-1, 512), rope.float()], dim=-1
    ).to(torch.bfloat16)


def test_selective_dequant_deduplicates_and_remaps_gpu():
    torch.manual_seed(7)
    device = torch.device("cuda")
    source = (torch.randn(12, 576, device=device) * 0.25).to(torch.bfloat16)
    packed = _pack_cache(source)
    # Two concatenated logical requests, deliberately mapped to non-contiguous
    # physical rows. The top-k rows contain overlap, duplicates and -1 padding.
    page_table = torch.tensor(
        [8, 2, 10, 4, 1, 11, 5, 8], device=device, dtype=torch.int32
    )
    topk = torch.tensor(
        [[0, 3, 3, -1], [6, 0, 7, -1], [4, 7, 4, 6]],
        device=device,
        dtype=torch.int32,
    )

    compact, remapped, used = dequantize_k_cache_paged_selective(
        packed,
        page_table,
        topk,
        max_unique_ratio=1.0,
        max_topk_ratio=2.0,
        min_tokens_saved=0,
        min_full_tokens=0,
    )

    assert used
    # Logical rows 0 and 7 alias physical slot 8 and are compacted together.
    unique_physical = torch.tensor([1, 4, 5, 8], device=device, dtype=torch.int32)
    expected = _reference_dequant(packed, unique_physical)
    torch.testing.assert_close(compact[:, 0], expected, rtol=0, atol=0)
    expected_remap = torch.tensor(
        [[3, 1, 1, -1], [2, 3, 3, -1], [0, 3, 0, 2]],
        device=device,
        dtype=torch.int32,
    )
    torch.testing.assert_close(remapped, expected_remap, rtol=0, atol=0)
    # Verify the remap, rather than only its chosen ordering, reconstructs every
    # selected logical KV row exactly.
    mask = topk >= 0
    selected_physical = page_table.index_select(0, topk[mask].long())
    torch.testing.assert_close(
        compact[:, 0].index_select(0, remapped[mask].long()),
        _reference_dequant(packed, selected_physical),
        rtol=0,
        atol=0,
    )


def test_selective_dequant_falls_back_when_savings_are_small_gpu():
    torch.manual_seed(11)
    device = torch.device("cuda")
    source = torch.randn(10, 576, device=device, dtype=torch.bfloat16)
    packed = _pack_cache(source)
    page_table = torch.randperm(10, device=device, dtype=torch.int32)
    topk = torch.tensor([[0, 1, 2], [3, 4, -1]], device=device, dtype=torch.int32)

    actual, actual_indices, used = dequantize_k_cache_paged_selective(
        packed,
        page_table,
        topk,
        max_unique_ratio=0.75,
        min_tokens_saved=0,
        min_full_tokens=131072,
    )
    expected = dequantize_k_cache_paged(packed, page_table)

    assert not used
    assert actual.data_ptr() != packed.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_indices, topk, rtol=0, atol=0)


def test_selective_dequant_falls_back_before_large_topk_dedup_gpu():
    torch.manual_seed(13)
    device = torch.device("cuda")
    source = torch.randn(16, 576, device=device, dtype=torch.bfloat16)
    packed = _pack_cache(source)
    page_table = torch.randperm(16, device=device, dtype=torch.int32)
    topk = torch.randint(0, 16, (4, 4), device=device, dtype=torch.int32)

    # This input has one top-k entry per full-prefix row, so the default
    # pre-deduplication gate must choose the full path.
    actual, actual_indices, used = dequantize_k_cache_paged_selective(
        packed,
        page_table,
        topk,
        min_full_tokens=0,
    )
    expected = dequantize_k_cache_paged(packed, page_table)

    assert not used
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_indices, topk, rtol=0, atol=0)


def test_selective_dequant_masks_invalid_indices_and_supplies_empty_sentinel_gpu():
    device = torch.device("cuda")
    source = torch.zeros(2, 576, device=device, dtype=torch.bfloat16)
    packed = _pack_cache(source)
    page_table = torch.tensor([0, 1], device=device, dtype=torch.int32)
    topk = torch.tensor([[-1, 2, -7]], device=device, dtype=torch.int32)

    compact, remapped, used = dequantize_k_cache_paged_selective(
        packed,
        page_table,
        topk,
        max_unique_ratio=1.0,
        max_topk_ratio=2.0,
        min_tokens_saved=0,
        min_full_tokens=0,
    )

    assert used
    assert compact.shape == (1, 1, 576)
    assert torch.count_nonzero(compact) == 0
    torch.testing.assert_close(remapped, torch.full_like(topk, -1))
