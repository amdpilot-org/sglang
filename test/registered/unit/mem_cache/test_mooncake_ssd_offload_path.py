import os
import tempfile
import types
import unittest
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fake_store_class():
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

        def is_exist(self, key):
            return int(key in self.objects)

        def remove(self, key, *args, **kwargs):
            self.objects.pop(key, None)
            return 0

    return FakeMooncakeDistributedStore


def _fake_modules(fake_store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls

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


def _make_storage_config(*, dp_rank, tp_rank, pp_rank, ssd_offload_path):
    return HiCacheStorageConfig(
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        tp_size=8,
        pp_rank=pp_rank,
        pp_size=2,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "check_server": False,
            "global_segment_size": 1024 * 1024,
            "enable_ssd_offload": True,
            "ssd_offload_path": ssd_offload_path,
        },
    )


def _make_store(storage_config):
    fake_store_cls = _fake_store_class()
    with patch.dict("sys.modules", _fake_modules(fake_store_cls)):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        MooncakeStore(storage_config)
    return fake_store_cls.instances[-1]


class TestMooncakeSsdOffloadPath(unittest.TestCase):
    def test_direct_linker_propagates_attention_dp_rank(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
            _direct_linker_storage_config,
        )

        params = types.SimpleNamespace(
            pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1
        )
        with (
            patch(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.get_parallel"
            ) as get_parallel,
            patch(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.get_model"
            ) as get_model,
        ):
            get_parallel.return_value.attn_dp_rank = 3
            get_model.return_value.model_path = "test"
            config = _direct_linker_storage_config(
                params=params,
                tp_rank=0,
                tp_size=1,
                rank_replicated=True,
                extra_config={},
            )

        self.assertEqual(config.dp_rank, 3)

    def test_dp_attention_clients_get_distinct_directories(self):
        with tempfile.TemporaryDirectory() as base:
            paths = {
                _make_store(
                    _make_storage_config(
                        dp_rank=dp_rank,
                        tp_rank=0,
                        pp_rank=0,
                        ssd_offload_path=base,
                    )
                ).setup_kwargs["ssd_offload_path"]
                for dp_rank in range(4)
            }

            self.assertEqual(
                paths,
                {os.path.join(base, f"rank_{rank}_0_0") for rank in range(4)},
            )
            self.assertTrue(all(os.path.isdir(path) for path in paths))

    def test_tp_and_pp_ranks_are_both_part_of_directory(self):
        with tempfile.TemporaryDirectory() as base:
            paths = {
                _make_store(
                    _make_storage_config(
                        dp_rank=1,
                        tp_rank=tp_rank,
                        pp_rank=pp_rank,
                        ssd_offload_path=base,
                    )
                ).setup_kwargs["ssd_offload_path"]
                for tp_rank, pp_rank in ((0, 0), (1, 0), (0, 1))
            }

            self.assertEqual(len(paths), 3)
            self.assertIn(os.path.join(base, "rank_1_0_1"), paths)

    def test_unconfigured_ssd_path_is_not_added(self):
        config = _make_storage_config(
            dp_rank=0, tp_rank=0, pp_rank=0, ssd_offload_path=None
        )
        client = _make_store(config)

        self.assertNotIn("ssd_offload_path", client.setup_kwargs)


if __name__ == "__main__":
    unittest.main()
