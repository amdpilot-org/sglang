import os
import sys
import tempfile
import types
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig


class FakeMooncakeDistributedStore:
    instances = []

    def __init__(self):
        self.setup_kwargs = None
        self.objects = {}
        type(self).instances.append(self)

    def setup(self, *args, **kwargs):
        self.setup_kwargs = kwargs
        return 0

    def put(self, key, value, *args, **kwargs):
        self.objects[key] = value
        return 0

    def get(self, key, *args, **kwargs):
        return self.objects.get(key)

    def is_exist(self, key, *args, **kwargs):
        return int(key in self.objects)

    def remove(self, key, *args, **kwargs):
        self.objects.pop(key, None)
        return 0


def fake_modules():
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = FakeMooncakeDistributedStore
    pool_host = types.ModuleType("sglang.srt.mem_cache.pool_host")
    pool_host.HostKVCache = type("HostKVCache", (), {})
    pool_host.HostTensorAllocator = type("HostTensorAllocator", (), {})
    pool_host_mla = types.ModuleType("sglang.srt.mem_cache.pool_host.mla")
    pool_host_mla.MLATokenToKVPoolHost = type("MLATokenToKVPoolHost", (), {})
    metrics = types.ModuleType("sglang.srt.observability.metrics_collector")
    metrics.StorageMetrics = type("StorageMetrics", (), {})
    return {
        "mooncake": mooncake,
        "mooncake.store": mooncake_store,
        "sglang.srt.mem_cache.pool_host": pool_host,
        "sglang.srt.mem_cache.pool_host.mla": pool_host_mla,
        "sglang.srt.observability.metrics_collector": metrics,
    }


def config(base, dp_rank=0, tp_rank=0, pp_rank=0):
    values = dict(
        tp_rank=tp_rank,
        tp_size=1,
        pp_rank=pp_rank,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="independent-review",
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "check_server": False,
            "global_segment_size": 1048576,
            "enable_ssd_offload": True,
            "ssd_offload_path": base,
        },
    )
    if "dp_rank" in HiCacheStorageConfig.__dataclass_fields__:
        values["dp_rank"] = dp_rank
    return HiCacheStorageConfig(**values)


def make_store(storage_config):
    with patch.dict(sys.modules, fake_modules()):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        MooncakeStore(storage_config)
    return FakeMooncakeDistributedStore.instances[-1]


print("hicache_storage_import=", sys.modules[HiCacheStorageConfig.__module__].__file__)
with tempfile.TemporaryDirectory() as base:
    paths = [
        make_store(config(base, dp_rank=rank)).setup_kwargs["ssd_offload_path"]
        for rank in (0, 1)
    ]
    print("dp_attention_paths=", paths)
    print("directories_exist=", [os.path.isdir(path) for path in paths])
    assert paths[0] != paths[1], (
        "DP-attention clients collide on the same SSD directory"
    )

with tempfile.TemporaryDirectory() as base:
    paths = [
        make_store(config(base, dp_rank=1, tp_rank=tp, pp_rank=pp)).setup_kwargs[
            "ssd_offload_path"
        ]
        for tp, pp in ((0, 0), (1, 0), (0, 1))
    ]
    print("topology_paths=", paths)
    assert len(set(paths)) == 3

try:
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
        _direct_linker_storage_config,
    )
except ImportError:
    _direct_linker_storage_config = None

if _direct_linker_storage_config is not None:
    params = types.SimpleNamespace(pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1)
    with tempfile.TemporaryDirectory() as base:
        direct_paths = []
        for dp_rank in (0, 1):
            with (
                patch(
                    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.get_parallel"
                ) as get_parallel,
                patch(
                    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.get_model"
                ) as get_model,
            ):
                get_parallel.return_value.attn_dp_rank = dp_rank
                get_model.return_value.model_path = "independent-review"
                direct_config = _direct_linker_storage_config(
                    params=params,
                    tp_rank=0,
                    tp_size=1,
                    rank_replicated=True,
                    extra_config=config(base).extra_config,
                )
            direct_paths.append(
                make_store(direct_config).setup_kwargs["ssd_offload_path"]
            )
        print("registered_direct_linker_paths=", direct_paths)
        assert direct_paths == [
            os.path.join(base, "rank_0_0_0"),
            os.path.join(base, "rank_1_0_0"),
        ]

print("PASS")
