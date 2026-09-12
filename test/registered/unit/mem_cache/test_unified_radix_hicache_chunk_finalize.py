"""Regression coverage for HiCache write-through after chunked prefill."""

import unittest
from array import array

import test_unified_radix_cache_unittest as base

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


class TestHiCacheChunkedFinalize(CustomTestCase):
    cfg = base.CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=60,
        kv_size=2048,
        max_context_len=2048,
    )
    _rid = 0
    _init_hicache = base.TestUnifiedRadixCacheKVEvents._init_hicache

    def _make_req(self, req_to_token_pool):
        req = Req(
            rid=str(self._rid),
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        type(self)._rid += 1
        req_to_token_pool.alloc([req])
        return req

    def _finish_after_chunk_boundary(
        self, cache, allocator, req_to_token_pool, *, expect_unbacked=True
    ):
        """Commit a chunk, then finish with no newly tracked Mamba checkpoint."""
        req = self._make_req(req_to_token_pool)
        tokens = list(range(1, 13))
        req.origin_input_ids = array("q", tokens)
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(tokens))
        kv_indices = allocator.alloc(len(tokens))
        self.assertIsNotNone(kv_indices)
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, len(tokens))), kv_indices
        )
        req.kv.kv_committed_len = len(tokens)
        req.kv.kv_allocated_len = len(tokens)
        req.last_node = cache.root_node_handle()
        req.kv.cache_protected_len = 0
        req.lock_receipt = DecLockRefParams()
        req.extra_key = None
        req.kv.mamba_last_track_seqlen = len(tokens)

        cache.cache_unfinished_req(req, chunked=True)
        self.assertIsNone(req.kv.mamba_last_track_seqlen)
        boundary_node = cache.resolve_node_handle(req.last_node)
        self.assertEqual(boundary_node.backuped, not expect_unbacked)

        req.output_ids = array("q", [2000])
        req.full_untruncated_fill_ids = array("q", tokens + [2000])
        req.set_extend_range(len(tokens), len(tokens) + 1)
        extra = allocator.alloc(1)
        self.assertIsNotNone(extra)
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(len(tokens), len(tokens) + 1)), extra
        )
        req.kv.kv_committed_len += 1
        req.kv.kv_allocated_len += 1

        cache.cache_finished_req(
            req, is_insert=True, kv_len_to_handle=req.effective_kv_committed_len()
        )
        return boundary_node

    def test_chunk_boundary_prefix_is_backed_up_on_finish(self):
        cache, allocator, req_to_token_pool = base.build_fixture(self.cfg)
        self._init_hicache(cache)
        cache.write_through_threshold = 1

        boundary_node = self._finish_after_chunk_boundary(
            cache, allocator, req_to_token_pool
        )

        self.assertTrue(boundary_node.backuped)
        cache.sanity_check()

    def test_selective_policy_counts_each_completed_request_once(self):
        cache, allocator, req_to_token_pool = base.build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_through_selective")
        cache.write_through_threshold = 2

        first = self._finish_after_chunk_boundary(cache, allocator, req_to_token_pool)
        self.assertFalse(first.backuped)
        self.assertEqual(first.hit_count, 1)

        second = self._finish_after_chunk_boundary(cache, allocator, req_to_token_pool)
        self.assertIs(second, first)
        self.assertTrue(second.backuped)
        self.assertEqual(second.hit_count, 2)
        cache.sanity_check()

    def test_write_back_does_not_eagerly_backup_on_finish(self):
        cache, allocator, req_to_token_pool = base.build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")

        boundary_node = self._finish_after_chunk_boundary(
            cache, allocator, req_to_token_pool
        )

        self.assertFalse(boundary_node.backuped)
        self.assertEqual(boundary_node.hit_count, 0)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
