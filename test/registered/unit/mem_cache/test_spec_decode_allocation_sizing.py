"""Regression tests for speculative decode KV reservation sizing."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache import allocation_sizing

page_aligned_decode_alloc_lens = allocation_sizing.page_aligned_decode_alloc_lens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _req(*, committed: int, allocated: int):
    return SimpleNamespace(
        kv=SimpleNamespace(
            kv_committed_len=committed,
            kv_allocated_len=allocated,
        )
    )


class TestSpecDecodeAllocationSizing(CustomTestCase):
    def test_synchronous_lengths_remove_overlap_double_reserve(self):
        reqs = [
            _req(committed=100, allocated=106),
            _req(committed=200, allocated=206),
        ]

        cur, nxt, needed = page_aligned_decode_alloc_lens(
            reqs,
            reserve=6,
            page_size=1,
            base_kv_lens=[106, 206],
        )

        self.assertEqual(cur, [106, 206])
        self.assertEqual(nxt, [112, 212])
        self.assertEqual(needed, 12)

    def test_synchronous_lengths_preserve_page_alignment(self):
        reqs = [_req(committed=95, allocated=104)]

        cur, nxt, needed = page_aligned_decode_alloc_lens(
            reqs,
            reserve=6,
            page_size=8,
            base_kv_lens=[101],
        )

        self.assertEqual(cur, [104])
        self.assertEqual(nxt, [112])
        self.assertEqual(needed, 8)

    def test_existing_committed_length_callers_are_unchanged(self):
        reqs = [_req(committed=101, allocated=104)]

        cur, nxt, needed = page_aligned_decode_alloc_lens(
            reqs,
            reserve=12,
            page_size=8,
        )

        self.assertEqual(cur, [104])
        self.assertEqual(nxt, [120])
        self.assertEqual(needed, 16)

    def test_encoder_image_tokens_are_excluded_from_target_length(self):
        req = _req(committed=10, allocated=10)
        req.seqlen = 22
        req.multimodal_inputs = SimpleNamespace(num_image_tokens=4)

        self.assertEqual(
            allocation_sizing.decode_seq_lens_from_reqs(
                [req], is_encoder_decoder=True
            ),
            [18],
        )


if __name__ == "__main__":
    unittest.main()
