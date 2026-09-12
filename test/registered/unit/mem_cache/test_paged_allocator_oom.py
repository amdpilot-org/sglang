"""Regression tests for paged allocator out-of-memory handling."""

from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_allocator(*, need_sort=False, size=8):
    return PagedTokenToKVPoolAllocator(
        size=size,
        page_size=4,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


@pytest.mark.parametrize("num_new_pages", [None, 3])
def test_extend_oom_does_not_launch_kernel(num_new_pages):
    allocator = _make_allocator()
    free_pages_before = allocator.free_pages.clone()
    prefix_lens = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([12], dtype=torch.int64)
    last_loc = torch.tensor([-1], dtype=torch.int64)

    with patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel") as kernel:
        result = allocator.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            extend_num_tokens=12,
            num_new_pages=num_new_pages,
        )

    assert result is None
    assert torch.equal(allocator.free_pages, free_pages_before)
    kernel.__getitem__.assert_not_called()


def test_decode_oom_does_not_launch_kernel():
    allocator = _make_allocator()
    free_pages_before = allocator.free_pages.clone()
    seq_lens = torch.tensor([5, 9, 13], dtype=torch.int64)
    last_loc = torch.tensor([3, 7, 11], dtype=torch.int64)

    with patch("sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel") as kernel:
        result = allocator.alloc_decode(seq_lens, seq_lens, last_loc)

    assert result is None
    assert torch.equal(allocator.free_pages, free_pages_before)
    kernel.__getitem__.assert_not_called()


@pytest.mark.parametrize("operation", ["extend", "decode"])
def test_exact_capacity_still_launches_and_consumes_pages(operation):
    allocator = _make_allocator()

    if operation == "extend":
        lens = torch.tensor([8], dtype=torch.int64)
        with patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel"
        ) as kernel:
            result = allocator.alloc_extend(
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                lens,
                lens,
                torch.tensor([-1], dtype=torch.int64),
                extend_num_tokens=8,
            )
    else:
        lens = torch.tensor([5, 9], dtype=torch.int64)
        with patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel"
        ) as kernel:
            result = allocator.alloc_decode(
                lens, lens, torch.tensor([3, 7], dtype=torch.int64)
            )

    assert result is not None
    assert len(allocator.free_pages) == 0
    kernel.__getitem__.assert_called_once_with((1,) if operation == "extend" else (2,))


def test_capacity_check_occurs_after_staged_pages_are_merged():
    allocator = _make_allocator(need_sort=True, size=12)
    allocated = allocator.alloc(8)
    allocator.free(allocated[:4])
    assert len(allocator.free_pages) == 1
    assert allocator.num_staged_pages == 1

    prefix_lens = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([8], dtype=torch.int64)
    last_loc = torch.tensor([-1], dtype=torch.int64)
    with patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel") as kernel:
        result = allocator.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            extend_num_tokens=8,
        )

    assert result is not None
    assert len(allocator.free_pages) == 0
    assert allocator.num_staged_pages == 0
    kernel.__getitem__.assert_called_once_with((1,))
