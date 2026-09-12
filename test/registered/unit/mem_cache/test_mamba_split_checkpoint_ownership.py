"""Regression coverage for Mamba radix split/checkpoint ownership.

These tests deliberately distinguish a structural KV match from a reusable
hybrid checkpoint.  A Mamba slot contains recurrent and convolution state at
one exact token depth, so the cache must not move a descendant slot to a new
parent created by splitting a compressed edge.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Allocator:
    def free(self, slots):
        pass


def _cache() -> MambaRadixCache:
    args = ServerArgs(model_path="dummy", page_size=1)
    args._mamba_cache_chunk_size = 64
    set_global_server_args_for_scheduler(args)
    return MambaRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=_Allocator(), mamba_ckpt_pool=None
            ),
            token_to_kv_pool_allocator=PagedTokenToKVPoolAllocator(
                size=512,
                page_size=1,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=None,
                need_sort=False,
            ),
            page_size=1,
        )
    )


def _insert(cache: MambaRadixCache, tokens, state_id: int) -> None:
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.arange(len(tokens), dtype=torch.int64),
            mamba_value=torch.tensor([state_id], dtype=torch.int64),
        )
    )


def _match(cache: MambaRadixCache, tokens):
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)), cow_mamba=False)
    )


class TestMambaSplitCheckpointOwnership(unittest.TestCase):
    def test_n_minus_one_split_cannot_relabel_leaf_state(self):
        cache = _cache()
        _insert(cache, [10, 20, 30], state_id=3)

        result = _match(cache, [10, 20])

        self.assertEqual(result.device_indices.numel(), 0)
        split = cache.root_node.children[10]
        leaf = split.children[30]
        self.assertIsNone(split.mamba_value)
        self.assertEqual(leaf.mamba_value.item(), 3)

    def test_exact_owned_checkpoint_makes_same_prefix_reusable(self):
        cache = _cache()
        _insert(cache, [10, 20], state_id=2)
        _insert(cache, [10, 20, 30], state_id=3)

        result = _match(cache, [10, 20])

        self.assertEqual(result.device_indices.tolist(), [0, 1])
        self.assertEqual(result.last_device_node.mamba_value.item(), 2)

    def test_arbitrary_depth_owned_checkpoint_survives_deeper_insert(self):
        cache = _cache()
        checkpoint = list(range(37))
        full = list(range(91))
        _insert(cache, checkpoint, state_id=37)
        _insert(cache, full, state_id=91)

        result = _match(cache, checkpoint + [999])

        self.assertEqual(result.device_indices.tolist(), list(range(37)))
        self.assertEqual(result.last_device_node.mamba_value.item(), 37)

    def test_divergence_before_first_owned_checkpoint_replays_from_root(self):
        cache = _cache()
        _insert(cache, list(range(64)), state_id=64)

        result = _match(cache, list(range(31)) + [999])

        self.assertEqual(result.device_indices.numel(), 0)
        self.assertIs(result.last_device_node, cache.root_node)


if __name__ == "__main__":
    unittest.main()
