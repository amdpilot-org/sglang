"""CPU-only regressions for external-cache committed KV boundaries."""

import importlib
import sys
import types
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Metadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _optional_dependency_stubs():
    lmcache_modules = {
        "lmcache": types.ModuleType("lmcache"),
        "lmcache.integration": types.ModuleType("lmcache.integration"),
        "lmcache.integration.sglang": types.ModuleType("lmcache.integration.sglang"),
        "lmcache.integration.sglang.multi_process_adapter": types.ModuleType(
            "lmcache.integration.sglang.multi_process_adapter"
        ),
        "lmcache.integration.sglang.sglang_adapter": types.ModuleType(
            "lmcache.integration.sglang.sglang_adapter"
        ),
        "lmcache.integration.sglang.utils": types.ModuleType(
            "lmcache.integration.sglang.utils"
        ),
    }
    lmcache_modules[
        "lmcache.integration.sglang.multi_process_adapter"
    ].LMCacheMPConnector = object
    adapter = lmcache_modules["lmcache.integration.sglang.sglang_adapter"]
    adapter.LMCacheLayerwiseConnector = object
    adapter.LoadMetadata = _Metadata
    adapter.StoreMetadata = _Metadata
    lmcache_modules["lmcache.integration.sglang.utils"].lmcache_get_config = MagicMock()

    flexkv_connector = types.ModuleType(
        "sglang.srt.mem_cache.storage.flexkv.flexkv_connector"
    )
    flexkv_connector.FlexKVConnector = object
    return {
        **lmcache_modules,
        "sglang.srt.mem_cache.storage.flexkv.flexkv_connector": flexkv_connector,
    }


class TestExternalCacheEffectiveKVLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._module_names = (
            "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache",
            "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache",
        )
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._module_names}
        for name in cls._module_names:
            sys.modules.pop(name, None)
        with patch.dict(sys.modules, _optional_dependency_stubs()):
            cls.lmc = importlib.import_module(cls._module_names[0])
            cls.flexkv = importlib.import_module(cls._module_names[1])

    @classmethod
    def tearDownClass(cls):
        for name, module in cls._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _make_cache_and_req(self, cache_cls, *, is_lmcache):
        cache = cache_cls.__new__(cache_cls)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int64).reshape(1, 32)
        )
        cache.inc_lock_ref = MagicMock()
        cache.dec_lock_ref = MagicMock()
        cache._node_lock = nullcontext()
        cache.store_stream = object()

        connector = MagicMock()
        if is_lmcache:
            cache._mode = self.lmc.LMCacheMode.MP
            cache._mp_load_back_markers = {}
        else:
            cache._inflight_store_nodes = {}
            connector.store_kv.return_value = -1
        cache.lmcache_connector = connector
        cache.flexkv_connector = connector

        req = SimpleNamespace(
            rid="req",
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[5, 6, 7, 8, 9],
            kv=SimpleNamespace(req_pool_idx=0, kv_committed_len=9),
            extra_key=None,
            cache_salt=None,
        )
        return cache, connector, req

    def _stored_token_ids_and_indices(self, cache_cls, is_lmcache, topk, limit):
        cache, connector, req = self._make_cache_and_req(
            cache_cls, is_lmcache=is_lmcache
        )
        match = SimpleNamespace(last_device_node=object())
        module = self.lmc if is_lmcache else self.flexkv
        with (
            patch.object(RadixCache, "cache_finished_req"),
            patch.object(RadixCache, "match_prefix", return_value=match),
            patch.object(
                module,
                "get_spec",
                return_value=SimpleNamespace(speculative_eagle_topk=topk),
            ),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
        ):
            cache.cache_finished_req(req, kv_len_to_handle=limit)

        if is_lmcache:
            metadata = connector.store_kv.call_args.args[0]
            return metadata.token_ids, metadata.kv_indices.tolist()
        kwargs = connector.store_kv.call_args.kwargs
        return kwargs["token_ids"], kwargs["kv_indices"].tolist()

    def test_external_stores_cap_normal_committed_length(self):
        for cache_cls, is_lmcache in (
            (self.lmc.LMCRadixCache, True),
            (self.flexkv.FlexKVRadixCache, False),
        ):
            with self.subTest(backend=cache_cls.__name__):
                token_ids, kv_indices = self._stored_token_ids_and_indices(
                    cache_cls, is_lmcache, topk=None, limit=4
                )
                self.assertEqual(token_ids, [1, 2, 3, 4])
                self.assertEqual(kv_indices, [0, 1, 2, 3])

    def test_external_stores_cap_speculative_length(self):
        for cache_cls, is_lmcache in (
            (self.lmc.LMCRadixCache, True),
            (self.flexkv.FlexKVRadixCache, False),
        ):
            with self.subTest(backend=cache_cls.__name__):
                token_ids, kv_indices = self._stored_token_ids_and_indices(
                    cache_cls, is_lmcache, topk=4, limit=4
                )
                self.assertEqual(token_ids, [1, 2, 3, 4])
                self.assertEqual(kv_indices, [0, 1, 2, 3])

    def test_larger_limit_does_not_extend_backend_committed_length(self):
        for cache_cls, is_lmcache in (
            (self.lmc.LMCRadixCache, True),
            (self.flexkv.FlexKVRadixCache, False),
        ):
            with self.subTest(backend=cache_cls.__name__):
                token_ids, kv_indices = self._stored_token_ids_and_indices(
                    cache_cls, is_lmcache, topk=4, limit=12
                )
                self.assertEqual(token_ids, list(range(1, 9)))
                self.assertEqual(kv_indices, list(range(8)))


if __name__ == "__main__":
    unittest.main()
