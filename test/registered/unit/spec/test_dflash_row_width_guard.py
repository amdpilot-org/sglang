"""Fail-fast coverage for DFLASH decode reservation row bounds."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestDFlashRowWidthGuard(CustomTestCase):
    def test_prepare_for_decode_rejects_reservation_past_row(self):
        req = SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=100, kv_committed_len=100),
            sampling_params=SimpleNamespace(top_k=1),
        )
        batch = SimpleNamespace(
            device=torch.device("cpu"),
            batch_size=lambda: 1,
            maybe_evict_swa=lambda: None,
            reqs=[req],
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.empty((1, 133), dtype=torch.int64)
            ),
        )
        draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))

        with get_context().override_server_args(speculative_num_draft_tokens=17):
            with self.assertRaisesRegex(
                AssertionError, "over-allocation .* exceeds req_to_token row width"
            ):
                draft_input.prepare_for_decode(batch)


if __name__ == "__main__":
    unittest.main()
