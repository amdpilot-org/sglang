"""Regression for paged decode refusal followed by a valid allocation."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.utils import get_num_new_pages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeDecodeKernel:
    def __getitem__(self, grid):
        def launch(seq_lens, last_loc, free_pages, out_indices, bs_upper, page_size):
            for index in range(len(seq_lens)):
                if index >= len(free_pages):
                    continue
                needs_page = page_size == 1 or seq_lens[index] % page_size == 1
                if needs_page:
                    out_indices[index] = free_pages[index] * page_size
                else:
                    out_indices[index] = last_loc[index] + 1

        return launch


class TestPagedAllocatorTransition(CustomTestCase):
    def test_decode_page_size_one_counts_every_token(self):
        seq_lens = torch.tensor([1, 2, 3], dtype=torch.int64)
        self.assertEqual(
            get_num_new_pages(seq_lens=seq_lens, page_size=1, decode=True), 3
        )

    def test_decode_refusal_preserves_capacity_for_valid_allocation(self):
        allocator = PagedTokenToKVPoolAllocator(
            size=8,
            page_size=1,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator.free_pages = torch.tensor([101, 202], dtype=torch.int64)
        free_pages_before = allocator.free_pages.clone()
        available_before = allocator.available_size()
        refused_lens = torch.tensor([1, 2, 3], dtype=torch.int64)
        refused_last_loc = torch.tensor([0, 1, 2], dtype=torch.int64)

        with patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel",
            FakeDecodeKernel(),
        ):
            refused = allocator.alloc_decode(
                refused_lens,
                refused_lens,
                refused_last_loc,
            )
            self.assertIsNone(refused)
            self.assertTrue(torch.equal(allocator.free_pages, free_pages_before))
            self.assertEqual(allocator.available_size(), available_before)

            valid_lens = torch.tensor([1], dtype=torch.int64)
            valid_last_loc = torch.tensor([0], dtype=torch.int64)
            valid = allocator.alloc_decode(
                valid_lens,
                valid_lens,
                valid_last_loc,
            )

        self.assertEqual(valid.tolist(), [101])
        self.assertEqual(allocator.available_size(), 1)
        self.assertEqual(allocator.free_pages.tolist(), [202])


if __name__ == "__main__":
    unittest.main()
