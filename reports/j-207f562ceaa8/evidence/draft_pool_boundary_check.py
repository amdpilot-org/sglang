"""Independent boundary checks for the fixed HiCache draft-pool dispatch."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_full_draft_pools,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


def hybrid(inner):
    wrapper = object.__new__(HybridLinearKVPool)
    wrapper.full_kv_pool = inner
    return wrapper


def main():
    # Boundary 1: the issue-shaped wrapper has no layer_num of its own. An
    # empty inner attention pool must return cleanly instead of raising.
    specs, entries = build_full_draft_pools(
        draft_kv_pool=hybrid(SimpleNamespace(layer_num=0)), tree_cache=None
    )
    assert specs == [] and entries == []

    # Boundaries 2 and 3: both a non-empty hybrid wrapper and a regular pool
    # must build a sidecar from the actual attention pool.
    publish(
        ServerArgs(model_path="dummy", hicache_mem_layout="page_first"),
        role="scheduler",
    )
    try:
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(logical_size=16), page_size=4
            )
        )
        for label, draft_pool, expected_device_pool in (
            (
                "nonempty_hybrid",
                hybrid(SimpleNamespace(layer_num=1, size=8)),
                None,
            ),
            ("nonempty_plain", SimpleNamespace(layer_num=1, size=8), None),
        ):
            expected_device_pool = (
                draft_pool.full_kv_pool
                if isinstance(draft_pool, HybridLinearKVPool)
                else draft_pool
            )
            host_pool = SimpleNamespace(layer_num=1)
            with (
                patch(
                    "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                    "_build_mha_mla_host_pool",
                    return_value=host_pool,
                ),
                patch(
                    "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                    "_get_allocator_type",
                    return_value="default",
                ),
            ):
                specs, entries = build_full_draft_pools(
                    draft_kv_pool=draft_pool, tree_cache=tree_cache
                )
            assert len(specs) == 1 and len(entries) == 1
            assert entries[0].device_pool is expected_device_pool
            print(f"{label}: PASS")
    finally:
        reset_context()

    print("empty_hybrid: PASS")


if __name__ == "__main__":
    main()
