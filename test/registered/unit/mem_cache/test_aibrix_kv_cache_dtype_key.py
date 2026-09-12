import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Status:
    value = 0

    def is_ok(self):
        return True


class _Manager:
    def __init__(self, config):
        self.config = config
        self.seen = []

    def exists(self, tenant, hashes):
        self.seen.append(hashes)
        return _Status()


class _Record:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.fixture
def aibrix_storage_class(monkeypatch):
    aibrix = types.ModuleType("aibrix_kvcache")
    aibrix.BaseKVCacheManager = _Manager
    aibrix.BlockHashes = lambda keys, page_size: (tuple(keys), page_size)
    aibrix.KVCacheBlockLayout = type(
        "KVCacheBlockLayout",
        (),
        {"NCLD": "NCLD", "__init__": lambda self, value: None},
    )
    aibrix.KVCacheBlockSpec = _Record
    aibrix.KVCacheConfig = _Record
    aibrix.KVCacheTensorSpec = _Record
    aibrix.ModelSpec = _Record
    common = types.ModuleType("aibrix_kvcache.common")
    logging_module = types.ModuleType("aibrix_kvcache.common.absl_logging")
    logging_module.log_every_n_seconds = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "aibrix_kvcache", aibrix)
    monkeypatch.setitem(sys.modules, "aibrix_kvcache.common", common)
    monkeypatch.setitem(
        sys.modules, "aibrix_kvcache.common.absl_logging", logging_module
    )

    source = (
        Path(__file__).parents[4]
        / "python/sglang/srt/mem_cache/storage/aibrix_kvcache/aibrix_kvcache_storage.py"
    )
    spec = importlib.util.spec_from_file_location("aibrix_storage_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AibrixKVCacheStorage


def _config(kv_cache_dtype):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="same/model",
        kv_cache_dtype=kv_cache_dtype,
    )


def _mem_pool():
    device_pool = SimpleNamespace(
        dtype=torch.float8_e4m3fnuz,
        layer_num=1,
        head_num=1,
        start_layer=0,
        end_layer=1,
        head_dim=64,
    )
    return SimpleNamespace(device_pool=device_pool, page_size=64)


def _seen_hash_input(storage_class, dtype):
    storage = storage_class(_config(dtype), _mem_pool())
    storage.batch_exists(["same_page"])
    return storage.kv_cache_manager.seen[-1]


def test_aibrix_block_hashes_distinguish_logical_fp8_formats(aibrix_storage_class):
    e4m3 = _seen_hash_input(aibrix_storage_class, "fp8_e4m3")
    e5m2 = _seen_hash_input(aibrix_storage_class, "fp8_e5m2")

    assert e4m3 != e5m2
    assert e4m3 == (("dtype_fp8_e4m3_same_page",), 64)
    assert e5m2 == (("dtype_fp8_e5m2_same_page",), 64)


def test_aibrix_none_dtype_keeps_legacy_hash_input(aibrix_storage_class):
    assert _seen_hash_input(aibrix_storage_class, None) == (("same_page",), 64)
