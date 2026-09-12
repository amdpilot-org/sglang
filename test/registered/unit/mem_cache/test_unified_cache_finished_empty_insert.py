"""Regression tests for empty finished-request inserts in UnifiedRadixCache."""

import unittest
from array import array

import torch
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need a GPU")
class TestCacheFinishedReqEmptyInsert(CustomTestCase):
    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=8,
        kv_size=64,
        max_context_len=64,
    )

    def _make_finished_req(self, req_to_token_pool, tokens, last_track_seqlen):
        req = Req(
            rid=f"empty-insert-{last_track_seqlen}-{len(tokens)}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req_to_token_pool.alloc([req])
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(tokens))
        req.kv.kv_committed_len = len(tokens)
        req.kv.kv_allocated_len = len(tokens)
        req.kv.cache_protected_len = 0
        req.swa_prefix_lock_released = False
        req.extra_key = None
        req.kv.mamba_last_track_seqlen = last_track_seqlen
        return req

    def _finish(
        self,
        cache,
        allocator,
        req_to_token_pool,
        tokens,
        track_seqlen,
        last_node=None,
        lock_result=None,
    ):
        req = self._make_finished_req(req_to_token_pool, tokens, track_seqlen)
        if tokens:
            kv_indices = allocator.alloc(len(tokens))
            self.assertIsNotNone(kv_indices)
            req_to_token_pool.write(
                (req.kv.req_pool_idx, slice(0, len(tokens))), kv_indices
            )
        req.last_node = last_node or cache.root_node_handle()
        if lock_result is not None:
            req.lock_receipt = lock_result.to_dec_params()
        cache.cache_finished_req(
            req, is_insert=True, kv_len_to_handle=req.effective_kv_committed_len()
        )
        return req

    def test_finished_req_without_track_boundary_inserts_nothing(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        tokens = [1, 2, 3]

        seed = [7, 7, 7]
        donor = Req(
            rid="seed-donor",
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req_to_token_pool.alloc([donor])
        seed_value = allocator.alloc(len(seed))
        self.assertIsNotNone(seed_value)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", seed)),
                value=seed_value,
                mamba_value=donor.kv.mamba_pool_idx.unsqueeze(0),
            )
        )
        matched = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seed))))
        self.assertNotEqual(matched.last_device_node, cache.root_node_handle())
        lock_result = cache.inc_lock_ref(matched.last_device_node)

        kv0 = allocator.available_size()
        m0 = req_to_token_pool.mamba_allocator.available_size()

        req = self._finish(
            cache,
            allocator,
            req_to_token_pool,
            tokens,
            None,
            last_node=matched.last_device_node,
            lock_result=lock_result,
        )

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertEqual(len(match.device_indices), 0)
        self.assertEqual(cache.mamba_evictable_size(), 1)
        self.assertEqual(req.last_node, matched.last_device_node)
        self.assertEqual(allocator.available_size(), kv0)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), m0)
        cache.sanity_check()

    def test_finished_req_with_track_boundary_still_inserts(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        tokens = [1, 2, 3]
        kv0 = allocator.available_size()

        self._finish(cache, allocator, req_to_token_pool, tokens, len(tokens))

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertEqual(len(match.device_indices), len(tokens))
        self.assertEqual(cache.mamba_evictable_size(), 1)
        self.assertEqual(allocator.available_size(), kv0 - len(tokens))
        cache.sanity_check()

    def test_finished_zero_length_request_is_noop(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        kv0 = allocator.available_size()
        m0 = req_to_token_pool.mamba_allocator.available_size()

        self._finish(cache, allocator, req_to_token_pool, [], None)

        self.assertEqual(allocator.available_size(), kv0)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), m0)
        cache.sanity_check()

    def test_insert_empty_key_is_pure_noop(self):
        cache, _, req_to_token_pool = build_fixture(self.cfg)
        m0 = req_to_token_pool.mamba_allocator.available_size()
        req = self._make_finished_req(req_to_token_pool, [9, 9, 9], None)

        result = cache.insert(
            InsertParams(
                key=RadixKey(array("q")),
                value=torch.tensor([], dtype=torch.int64),
                mamba_value=req.kv.mamba_pool_idx.unsqueeze(0),
            )
        )

        self.assertEqual(result.prefix_len, 0)
        self.assertTrue(result.mamba_exist)
        self.assertIsNone(result.last_device_node)
        req_to_token_pool.free_mamba_cache(req)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), m0)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
