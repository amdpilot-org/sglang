"""Actual-cache ownership probe with independent GPU index references."""

from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class Allocator:
    def free(self, slots):
        pass


def cache():
    args = ServerArgs(model_path="dummy", page_size=1)
    args._mamba_cache_chunk_size = 64
    set_global_server_args_for_scheduler(args)
    return MambaRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=Allocator(), mamba_ckpt_pool=None
            ),
            token_to_kv_pool_allocator=PagedTokenToKVPoolAllocator(
                size=512,
                page_size=1,
                dtype=torch.bfloat16,
                device="cuda",
                kvcache=None,
                need_sort=False,
            ),
            page_size=1,
        )
    )


def insert(tree, tokens, state_id):
    tree.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.arange(len(tokens), device="cuda", dtype=torch.int64),
            mamba_value=torch.tensor([state_id], device="cuda", dtype=torch.int64),
        )
    )


def match(tree, tokens):
    return tree.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)), cow_mamba=False)
    ).device_indices


print("device", torch.cuda.get_device_name(0))
print("torch", torch.__version__)

tree = cache()
insert(tree, [10, 20, 30], 3)
print("leaf_depth_3_n_minus_1_count", match(tree, [10, 20]).numel())

tree = cache()
insert(tree, [10, 20], 2)
insert(tree, [10, 20, 30], 3)
actual = match(tree, [10, 20])
reference = torch.arange(2, device="cuda", dtype=torch.int64)
print("exact_depth_2_count", actual.numel())
print("exact_depth_2_reference_equal", torch.equal(actual, reference))

tree = cache()
insert(tree, list(range(37)), 37)
insert(tree, list(range(91)), 91)
actual = match(tree, list(range(37)) + [999])
reference = torch.arange(37, device="cuda", dtype=torch.int64)
print("arbitrary_depth_37_count", actual.numel())
print("arbitrary_depth_37_reference_equal", torch.equal(actual, reference))

tree = cache()
insert(tree, list(range(64)), 64)
print(
    "divergence_before_first_checkpoint_count",
    match(tree, list(range(31)) + [999]).numel(),
)
