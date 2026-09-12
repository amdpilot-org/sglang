"""Exercise MambaRadixCache with real ROCm tensors on the assigned GPU."""

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


def insert(cache, tokens, state_id):
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.arange(len(tokens), device="cuda", dtype=torch.int64),
            mamba_value=torch.tensor([state_id], device="cuda", dtype=torch.int64),
        )
    )


def match(cache, tokens):
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))


args = ServerArgs(model_path="dummy", page_size=1)
args._mamba_cache_chunk_size = 64
set_global_server_args_for_scheduler(args)
cache = MambaRadixCache(
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

insert(cache, [10, 20, 30], 3)
short = match(cache, [10, 20])
full = match(cache, [10, 20, 30])
print("device", torch.cuda.get_device_name(0))
print("short_n_minus_1_count", short.device_indices.numel())
print("full_count", full.device_indices.numel())
print("full_reference_equal", torch.equal(full.device_indices, torch.arange(3, device="cuda")))

cache.reset()
insert(cache, list(range(64)), 64)
insert(cache, list(range(96)), 96)
long = match(cache, list(range(95)))
reference = torch.arange(64, device="cuda", dtype=torch.int64)
print("checkpointed_long_count", long.device_indices.numel())
print("checkpointed_long_reference_equal", torch.equal(long.device_indices, reference))
print("checkpointed_long_branch", long.mamba_branching_seqlen)
