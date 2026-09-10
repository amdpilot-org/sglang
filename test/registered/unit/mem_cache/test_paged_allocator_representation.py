"""Representation contract tests for the paged allocator kernels."""

from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _make_allocator() -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        size=16,
        page_size=4,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )


def _extend_inputs():
    return (
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([8], dtype=torch.int64),
        torch.tensor([-1], dtype=torch.int64),
    )


def _decode_inputs():
    return (
        torch.tensor([5, 9], dtype=torch.int64),
        torch.tensor([3, 7], dtype=torch.int64),
    )


@pytest.mark.parametrize("operation", ["extend", "decode"])
def test_noncontiguous_free_pages_fail_before_launch(operation):
    allocator = _make_allocator()
    backing = torch.tensor([1, -1, 3, -1], dtype=torch.int64)
    allocator.free_pages = backing[::2]
    kernel_name = f"alloc_{operation}_kernel"

    with patch(f"sglang.srt.mem_cache.allocator.paged.{kernel_name}") as kernel:
        with pytest.raises(ValueError, match="free_pages must be contiguous"):
            if operation == "extend":
                prefix_lens, seq_lens, last_loc = _extend_inputs()
                allocator.alloc_extend(
                    prefix_lens,
                    prefix_lens,
                    seq_lens,
                    seq_lens,
                    last_loc,
                    extend_num_tokens=8,
                )
            else:
                seq_lens, last_loc = _decode_inputs()
                allocator.alloc_decode(seq_lens, seq_lens, last_loc)

    kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize("operation", ["extend", "decode"])
def test_packed_free_pages_fail_before_launch(operation):
    allocator = _make_allocator()
    allocator.free_pages = torch.tensor([[1], [3]], dtype=torch.int64)
    kernel_name = f"alloc_{operation}_kernel"

    with patch(f"sglang.srt.mem_cache.allocator.paged.{kernel_name}") as kernel:
        with pytest.raises(ValueError, match="free_pages must be a 1-D tensor"):
            if operation == "extend":
                prefix_lens, seq_lens, last_loc = _extend_inputs()
                allocator.alloc_extend(
                    prefix_lens,
                    prefix_lens,
                    seq_lens,
                    seq_lens,
                    last_loc,
                    extend_num_tokens=8,
                )
            else:
                seq_lens, last_loc = _decode_inputs()
                allocator.alloc_decode(seq_lens, seq_lens, last_loc)

    kernel.__getitem__.assert_not_called()


def test_noncontiguous_input_tensor_fails_before_launch():
    allocator = _make_allocator()
    backing = torch.tensor([0, -1, 1, -2], dtype=torch.int64)
    prefix_lens = backing[::2]
    seq_lens = torch.tensor([8], dtype=torch.int64)
    last_loc = torch.tensor([-1], dtype=torch.int64)

    with patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel") as kernel:
        with pytest.raises(ValueError, match="prefix_lens must be contiguous"):
            allocator.alloc_extend(
                prefix_lens,
                prefix_lens,
                seq_lens,
                seq_lens,
                last_loc,
                extend_num_tokens=8,
            )

    kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize("operation", ["extend", "decode"])
def test_wrong_free_pages_dtype_fails_before_launch(operation):
    allocator = _make_allocator()
    allocator.free_pages = torch.tensor([1, 3], dtype=torch.int32)
    kernel_name = f"alloc_{operation}_kernel"

    with patch(f"sglang.srt.mem_cache.allocator.paged.{kernel_name}") as kernel:
        with pytest.raises(ValueError, match="free_pages must have dtype"):
            if operation == "extend":
                prefix_lens, seq_lens, last_loc = _extend_inputs()
                allocator.alloc_extend(
                    prefix_lens,
                    prefix_lens,
                    seq_lens,
                    seq_lens,
                    last_loc,
                    extend_num_tokens=8,
                )
            else:
                seq_lens, last_loc = _decode_inputs()
                allocator.alloc_decode(seq_lens, seq_lens, last_loc)

    kernel.__getitem__.assert_not_called()
