from __future__ import annotations

import argparse
from array import array

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cpp_radix_tree.radix_tree import RadixTreeCpp
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp


class ReqToTokenPool:
    def __init__(self, row):
        self.req_to_token = row.unsqueeze(0)

    def write(self, indices, values):
        self.req_to_token[indices] = values


class Req:
    def __init__(self, token_ids, last_node):
        self._token_ids = token_ids
        self.kv = ReqKvInfo(req_pool_idx=0, cache_protected_len=0)
        self.extra_key = None
        self.cache_salt = None
        self.priority = 0
        self.last_node = last_node
        self.prefix_indices = torch.empty(0, dtype=torch.int64)

    def get_fill_ids(self):
        return self._token_ids


def run(path, device, no_clone):
    page_size = 64
    allocator = PagedTokenToKVPoolAllocator(
        size=page_size * 8, page_size=page_size, dtype=torch.float16,
        device=device, kvcache=None, need_sort=False,
    )
    token_ids = array("q", range(page_size))
    tree_indices = allocator.alloc(page_size)
    request_indices = allocator.alloc(page_size)
    pool = ReqToTokenPool(request_indices.clone())

    if path == "python":
        cache = RadixCache.create_simulated(page_size=page_size, mock_allocator=allocator)
        cache.req_to_token_pool = pool
        cache.insert(InsertParams(key=RadixKey(token_ids), value=tree_indices))
        root = cache.root_node
    else:
        cache = RadixCacheCpp.__new__(RadixCacheCpp)
        cache.disable = False
        cache.enable_write_cancel = False
        cache.token_to_kv_pool_allocator = allocator
        cache.device = allocator.device
        cache.req_to_token_pool = pool
        cache.page_size = page_size
        cache.kv_cache = None
        cache.cache_controller = None
        cache.tree = RadixTreeCpp(False, None, page_size, 2)
        cache._insert(RadixKey(token_ids), tree_indices.to(torch.int64, copy=True))
        _, _, root, _ = cache.tree.match_prefix([])

    req = Req(token_ids, root)
    old = BaseTokenToKVPoolAllocator._copy_for_free_group
    if no_clone:
        BaseTokenToKVPoolAllocator._copy_for_free_group = staticmethod(lambda x: x)
    try:
        allocator.free_group_begin()
        cache.cache_unfinished_req(req, chunked=True)
        allocator.free_group_end()
    finally:
        BaseTokenToKVPoolAllocator._copy_for_free_group = old

    free_pages = set(allocator.free_pages.cpu().tolist())
    request_page = int((request_indices[0] // page_size).cpu())
    tree_page = int((tree_indices[0] // page_size).cpu())
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
    result = {
        "path": path, "device": device, "clone_enabled": not no_clone,
        "request_page_freed": request_page in free_pages,
        "tree_page_freed": tree_page in free_pages,
        "row_is_tree": torch.equal(pool.req_to_token[0], tree_indices),
        "match_is_tree": torch.equal(match.device_indices, tree_indices),
    }
    print(result)
    assert result["request_page_freed"] and not result["tree_page_freed"]
    assert result["row_is_tree"] and result["match_is_tree"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", choices=("python", "cpp"))
    p.add_argument("device", choices=("cpu", "cuda"))
    p.add_argument("clone", choices=("clone", "no_clone"))
    a = p.parse_args()
    run(a.path, a.device, a.clone == "no_clone")
