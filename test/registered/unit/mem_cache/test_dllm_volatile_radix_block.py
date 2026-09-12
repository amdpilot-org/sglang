import unittest
from array import array

import torch

from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci


PAGE_SIZE = 32


class _DllmReq(ReqDllmMixin):
    def __init__(self, fill_ids, incomplete_ids=()):
        self.full_untruncated_fill_ids = array("q", fill_ids)
        self.extend_range = Range(0, len(fill_ids))
        self.dllm_incomplete_ids = array("q", incomplete_ids)
        self.kv = ReqKvInfo(req_pool_idx=0)
        self.last_node = None
        self.extra_key = None
        self.cache_salt = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.priority = 0
        self.session = None
        self.swa_uuid_for_lock = None
        self.skip_lock_node_ids = {}
        self.kv_rotation_base = None
        self.swa_prefix_lock_released = False
        self.lock_receipt = None

    def get_fill_ids(self):
        return self.full_untruncated_fill_ids[: self.extend_range.end]


def _make_pools():
    kv_pool = MHATokenToKVPool(
        size=1024,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=1024,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=2, max_context_len=256, device="cpu", enable_memory_saver=False
    )
    return allocator, req_pool


def _params(allocator, req_pool, **kwargs):
    return CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=PAGE_SIZE,
        eviction_policy="lru",
        enable_kv_cache_events=False,
        **kwargs,
    )


def _distinct_pages(cache, value_of):
    pages = set()
    stack = [cache.root_node]
    while stack:
        node = stack.pop()
        if node is not cache.root_node:
            pages.update(index // PAGE_SIZE for index in value_of(node).tolist())
        stack.extend(node.children.values())
    return pages


class DllmVolatileRadixBlockTest(unittest.TestCase):
    def test_cacheable_fill_ids_exclude_only_incomplete_block(self):
        stable = list(range(64))
        block = list(range(900, 932))
        req = _DllmReq(stable + block, block)
        self.assertEqual(list(req.get_cacheable_fill_ids()), stable)

        req.dllm_incomplete_ids = array("q")
        self.assertEqual(list(req.get_cacheable_fill_ids()), stable + block)

    def test_empty_incomplete_block_leaves_short_request_unchanged(self):
        req = _DllmReq([1, 2, 3])
        self.assertEqual(list(req.get_cacheable_fill_ids()), [1, 2, 3])

    def test_classic_cache_does_not_claim_volatile_page_twice(self):
        allocator, req_pool = _make_pools()
        cache = RadixCache(_params(allocator, req_pool))
        self._assert_no_duplicate_ownership(
            cache, allocator, req_pool.req_to_token, lambda node: node.value
        )

    def test_unified_cache_does_not_claim_volatile_page_twice(self):
        allocator, req_pool = _make_pools()
        cache = UnifiedRadixCache(
            _params(allocator, req_pool, tree_components=(ComponentType.FULL,))
        )
        self._assert_no_duplicate_ownership(
            cache,
            allocator,
            req_pool.req_to_token,
            lambda node: node.component(ComponentType.FULL).value,
        )

    def _assert_no_duplicate_ownership(self, cache, allocator, req_to_token, value_of):
        stable = list(range(64))
        req = _DllmReq(stable)
        req.last_node = cache.root_node_handle()
        if isinstance(cache, UnifiedRadixCache):
            req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        req_to_token[0, :96] = allocator.alloc(96).to(torch.int64)

        for block in (list(range(900, 932)), list(range(800, 832))):
            req.full_untruncated_fill_ids = array("q", stable + block)
            req.extend_range = Range(0, 96)
            req.dllm_incomplete_ids = array("q", block)
            cache.cache_unfinished_req(req)

        distinct_tokens = len(_distinct_pages(cache, value_of)) * PAGE_SIZE
        accounted_tokens = cache.evictable_size() + cache.protected_size()
        self.assertLessEqual(accounted_tokens, distinct_tokens)


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


if __name__ == "__main__":
    unittest.main()
