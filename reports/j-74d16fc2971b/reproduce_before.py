"""Deterministic reproduction of issue 18262's AITER budget overcommit."""

import os
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator


GIB = 1 << 30
AVAILABLE_BYTES = 219_721_119_039
CELL_SIZE = 98_304
CONTEXT_LEN = 262_144
NUM_HEADS = 32
HEAD_DIM = 128
PARTITION_SIZE = 256


class LinearPoolConfigurator:
    _cell_size = CELL_SIZE

    def finalize_with_max_running_requests(self, config):
        return config


def main():
    runner = object.__new__(KVCacheConfigurator)
    runner.model_config = SimpleNamespace(
        context_len=CONTEXT_LEN,
        num_attention_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    runner.ps = SimpleNamespace(attn_dp_size=1)
    with (
        patch.object(
            KVCacheConfigurator,
            "_profile_available_bytes",
            lambda self, _preload: AVAILABLE_BYTES,
        ),
        patch.object(
            KVCacheConfigurator,
            "_needs_aiter_workspace_reservation",
            lambda self: os.environ.get("SGLANG_REPRO_BEFORE") != "1",
        ),
        patch.object(
            KVCacheConfigurator,
            "config_from_budget",
            lambda self, budget: SimpleNamespace(
                max_total_num_tokens=budget // CELL_SIZE,
                max_running_requests=None,
                mem_fraction_static=None,
            ),
        ),
        patch.object(
            KVCacheConfigurator,
            "resolve_max_num_reqs",
            lambda self, tokens, **_kwargs: max(
                min(int(tokens / CONTEXT_LEN * 512), 4096), 2048
            ),
        ),
        patch(
            "sglang.srt.model_executor.pool_configurator.create_memory_pool_configurator",
            return_value=LinearPoolConfigurator(),
        ),
        patch(
            "sglang.srt.mem_cache.kv_cache_configurator.get_schedule",
            return_value=SimpleNamespace(mem_fraction_static=0.9),
        ),
        patch(
            "sglang.srt.mem_cache.kv_cache_configurator.get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1),
        ),
    ):
        config = runner._resolve_memory_pool_config(256)

    partitions = (CONTEXT_LEN + PARTITION_SIZE - 1) // PARTITION_SIZE
    bytes_per_request = NUM_HEADS * partitions * (HEAD_DIM * 4 + 8)
    workspace_bytes = config.max_running_requests * bytes_per_request
    total_bytes = config.max_total_num_tokens * CELL_SIZE + workspace_bytes

    print(f"available_bytes={AVAILABLE_BYTES}")
    print(f"kv_tokens={config.max_total_num_tokens}")
    print(f"max_running_requests={config.max_running_requests}")
    print(f"workspace_bytes={workspace_bytes}")
    print(f"total_bytes={total_bytes}")
    print(f"overcommit_bytes={total_bytes - AVAILABLE_BYTES}")
    assert total_bytes <= AVAILABLE_BYTES, "AITER workspace is outside the KV budget"


if __name__ == "__main__":
    main()
