"""Device-level lifecycle coverage for explicit radix-cache insertion opt-out."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import (
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


class TestSkipCacheInsertDeviceLifecycle(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.kv_pool = MHATokenToKVPool(
            size=16,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
        )
        self.req_to_token_pool = ReqToTokenPool(
            size=4,
            max_context_len=16,
            device=self.device,
            enable_memory_saver=False,
        )
        self.allocator = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float16,
            device=self.device,
            kvcache=self.kv_pool,
            need_sort=False,
        )
        self.cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
            )
        )

    def tearDown(self):
        self.cache.reset()
        del self.cache
        del self.allocator
        del self.req_to_token_pool
        del self.kv_pool

    def _make_request(self, tokens, *, skip_cache_insert, cache_salt):
        req_pool_idx = self.req_to_token_pool.alloc_rows(1)[0]
        kv_indices = self.allocator.alloc(len(tokens))
        self.req_to_token_pool.req_to_token[req_pool_idx, : len(tokens)] = (
            kv_indices.to(torch.int32)
        )

        key_buffer, value_buffer = self.kv_pool.get_kv_buffer(0)
        positions = torch.arange(
            len(tokens), device=self.device, dtype=torch.float16
        ).view(-1, 1)
        key_buffer[kv_indices, 0, :] = positions + 0.125
        value_buffer[kv_indices, 0, :] = 1 / (positions + 1)

        request = SimpleNamespace(
            skip_radix_cache_insert=skip_cache_insert,
            kv=ReqKvInfo(
                req_pool_idx=req_pool_idx,
                cache_protected_len=0,
                kv_committed_len=len(tokens),
                kv_allocated_len=len(tokens),
            ),
            origin_input_ids=array("q", tokens),
            output_ids=array("q"),
            extra_key=None,
            cache_salt=cache_salt,
            last_node=self.cache.root_node,
            prefix_indices=torch.empty(0, dtype=torch.int64, device=self.device),
            get_fill_ids=lambda: array("q", tokens),
            effective_kv_committed_len=lambda: request.kv.kv_committed_len,
        )
        return request, kv_indices

    def _release(self, request, *, is_insert=True):
        with (
            patch(
                "sglang.srt.mem_cache.common.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch(
                "sglang.srt.mem_cache.common.get_serving",
                return_value=SimpleNamespace(strip_thinking_cache=False),
            ),
        ):
            release_kv_cache(
                request,
                self.cache,
                is_insert=is_insert,
            )

    def _match(self, tokens, cache_salt):
        return self.cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    array("q", tokens),
                    cache_salt=cache_salt,
                )
            )
        ).device_indices

    def _attention(self, kv_indices):
        query = torch.ones((1, 1, 8), device=self.device, dtype=torch.float32)
        key_buffer, value_buffer = self.kv_pool.get_kv_buffer(0)
        keys = key_buffer[kv_indices, 0, :].unsqueeze(0).unsqueeze(0).to(torch.float32)
        values = (
            value_buffer[kv_indices, 0, :].unsqueeze(0).unsqueeze(0).to(torch.float32)
        )
        return torch.nn.functional.scaled_dot_product_attention(query, keys, values)[
            0, 0
        ]

    def test_opted_in_request_caches_and_preserves_attention_output(self):
        tokens = [1, 2, 3, 4, 5, 6]
        request, _ = self._make_request(
            tokens,
            skip_cache_insert=False,
            cache_salt="tenant-a",
        )

        maybe_cache_unfinished_req(request, self.cache)
        expected_output = self._attention(request.prefix_indices)
        self._release(request)

        cached_indices = self._match(tokens, "tenant-a")
        self.assertEqual(cached_indices.numel(), 6)
        self.assertEqual(self.cache.total_size(), 6)
        self.assertEqual(self.allocator.available_size(), 10)
        self.assertEqual(self.req_to_token_pool.available_size(), 4)
        self.assertIsNone(request.kv.req_pool_idx)
        self.assertEqual(request.kv.kv_allocated_len, 0)
        torch.testing.assert_close(self._attention(cached_indices), expected_output)
        self.assertEqual(self._match(tokens, "tenant-b").numel(), 0)

    def test_opted_out_request_releases_slots_without_radix_entry(self):
        tokens = [7, 8, 9, 10, 11, 12]
        request, kv_indices = self._make_request(
            tokens,
            skip_cache_insert=True,
            cache_salt="tenant-b",
        )

        maybe_cache_unfinished_req(request, self.cache)
        torch.testing.assert_close(request.prefix_indices, kv_indices)
        self.assertEqual(self.cache.total_size(), 0)
        self._release(request)

        self.assertEqual(self._match(tokens, "tenant-b").numel(), 0)
        self.assertEqual(self.cache.total_size(), 0)
        self.assertEqual(self.allocator.available_size(), 16)
        self.assertEqual(self.req_to_token_pool.available_size(), 4)
        self.assertIsNone(request.kv.req_pool_idx)
        self.assertEqual(request.kv.kv_allocated_len, 0)
        self.assertTrue(torch.isin(kv_indices, self.allocator.free_pages).all().item())

    def test_interrupted_opted_out_request_releases_allocated_slots(self):
        tokens = [13, 14, 15, 16, 17, 18]
        request, kv_indices = self._make_request(
            tokens,
            skip_cache_insert=True,
            cache_salt="tenant-c",
        )

        self._release(request, is_insert=False)

        self.assertEqual(self._match(tokens, "tenant-c").numel(), 0)
        self.assertEqual(self.cache.total_size(), 0)
        self.assertEqual(self.allocator.available_size(), 16)
        self.assertEqual(self.req_to_token_pool.available_size(), 4)
        self.assertIsNone(request.kv.req_pool_idx)
        self.assertEqual(request.kv.kv_allocated_len, 0)
        self.assertTrue(torch.isin(kv_indices, self.allocator.free_pages).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
