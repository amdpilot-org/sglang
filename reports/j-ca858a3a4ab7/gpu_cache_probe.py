from array import array

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


assert torch.cuda.is_available(), "assigned ROCm GPU unavailable"
props = torch.cuda.get_device_properties(0)
args = ServerArgs(model_path="dummy", page_size=1)
args._mamba_cache_chunk_size = 64
set_global_server_args_for_scheduler(args)
allocator = PagedTokenToKVPoolAllocator(
    size=4096,
    page_size=1,
    dtype=torch.bfloat16,
    device="cuda",
    kvcache=None,
    need_sort=False,
)
cache = MambaRadixCache(
    CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
    )
)


def insert(ids):
    value = torch.arange(len(ids), dtype=torch.int64, device="cuda")
    state = torch.arange(8, dtype=torch.float32, device="cuda").reshape(1, 8)
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", ids)), value=value, mamba_value=state
        )
    )


def match(ids):
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", ids))))


insert([1, 2, 3])
zero = match([1, 2])
full = match([1, 2, 3])
assert zero.device_indices.numel() == 0
assert torch.equal(full.device_indices, torch.arange(3, device="cuda"))

cache.reset()
insert(list(range(64)))
insert(list(range(94)))
hit = match(list(range(93)))
expected = torch.arange(64, device="cuda")
assert torch.equal(hit.device_indices, expected)

print(
    {
        "gpu": props.name,
        "rocm": torch.version.hip,
        "baseline_split_hit": zero.device_indices.numel(),
        "full_leaf_hit": full.device_indices.numel(),
        "checkpointed_split_hit": hit.device_indices.numel(),
        "indices_equal_reference": bool(torch.equal(hit.device_indices, expected)),
    }
)
