"""Regression tests for DFLASH/DSPARK req_to_token row sizing."""

import unittest

from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestDFlashAllocationSizing(CustomTestCase):
    def _assert_row_covers_reserve(self, algorithm: str, page_size: int):
        with (
            get_context().override_server_args(
                speculative_algorithm=algorithm,
                speculative_num_steps=None,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=17,
                page_size=page_size,
            ),
            get_parallel().override(attn_dcp_size=1),
        ):
            reserve = get_alloc_reserve_per_decode()
            extra = get_req_to_token_extra_context_len()
            self.assertGreaterEqual(extra, reserve + page_size - 1)

    def test_dspark_page_size_one_row_covers_decode_reserve(self):
        self._assert_row_covers_reserve("DSPARK", page_size=1)

    def test_dflash_page_size_one_row_covers_decode_reserve(self):
        self._assert_row_covers_reserve("DFLASH", page_size=1)

    def test_dflash_larger_page_row_covers_alignment_overshoot(self):
        self._assert_row_covers_reserve("DFLASH", page_size=16)

    def test_non_spec_page_size_one_headroom_is_unchanged(self):
        with (
            get_context().override_server_args(
                speculative_algorithm=None,
                speculative_num_draft_tokens=None,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
        ):
            self.assertEqual(get_req_to_token_extra_context_len(), 4)


if __name__ == "__main__":
    unittest.main()
