"""Independent actual-cache probe for candidate 6eaf6c1 and the prepared base."""

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


def make_cache():
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


def insert(cache, tokens, state_id):
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.arange(len(tokens), device="cuda", dtype=torch.int64),
            mamba_value=torch.tensor([state_id], device="cuda", dtype=torch.int64),
        )
    )


def match(cache, tokens):
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)), cow_mamba=False)
    )


print("device", torch.cuda.get_device_name(0))
print("torch", torch.__version__)

cache = make_cache()
insert(cache, [10, 20, 30], 3)
short = match(cache, [10, 20])
print("leaf_depth_3_n_minus_1_count", short.device_indices.numel())

cache = make_cache()
insert(cache, [10, 20], 2)
insert(cache, [10, 20, 30], 3)
exact = match(cache, [10, 20])
print("exact_depth_2_count", exact.device_indices.numel())
print(
    "exact_depth_2_reference_equal",
    torch.equal(exact.device_indices, torch.arange(2, device="cuda")),
)

cache = make_cache()
checkpoint = list(range(37))
insert(cache, checkpoint, 37)
insert(cache, list(range(91)), 91)
arbitrary = match(cache, checkpoint + [999])
print("arbitrary_depth_37_count", arbitrary.device_indices.numel())
print(
    "arbitrary_depth_37_reference_equal",
    torch.equal(arbitrary.device_indices, torch.arange(37, device="cuda")),
)

cache = make_cache()
insert(cache, list(range(64)), 64)
early = match(cache, list(range(31)) + [999])
print("divergence_before_first_checkpoint_count", early.device_indices.numel())
