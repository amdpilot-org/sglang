import sys
import types
from types import SimpleNamespace

import torch


class Status:
    def is_ok(self): return True
    value = 0


class Manager:
    def __init__(self, config): self.config = config; self.seen = []
    def exists(self, tenant, hashes): self.seen.append(hashes); return Status()


class Record:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


root = types.ModuleType("aibrix_kvcache")
root.BaseKVCacheManager = Manager
root.BlockHashes = lambda keys, page_size: (tuple(keys), page_size)
root.KVCacheBlockLayout = type("KVCacheBlockLayout", (), {"NCLD": "NCLD", "__init__": lambda self, x: None})
root.KVCacheBlockSpec = Record
root.KVCacheConfig = Record
root.KVCacheTensorSpec = Record
root.ModelSpec = Record
common = types.ModuleType("aibrix_kvcache.common")
logging_mod = types.ModuleType("aibrix_kvcache.common.absl_logging")
logging_mod.log_every_n_seconds = lambda *args, **kwargs: None
sys.modules.update({
    "aibrix_kvcache": root,
    "aibrix_kvcache.common": common,
    "aibrix_kvcache.common.absl_logging": logging_mod,
})

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.aibrix_kvcache.aibrix_kvcache_storage import AibrixKVCacheStorage


device_pool = SimpleNamespace(
    dtype=torch.float8_e4m3fnuz, layer_num=1, head_num=1,
    start_layer=0, end_layer=1, head_dim=64,
)
mem_pool = SimpleNamespace(device_pool=device_pool, page_size=64)


def config(dtype):
    return HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
        enable_storage_metrics=False, is_page_first_layout=False,
        model_name="same/model", kv_cache_dtype=dtype,
    )


a = AibrixKVCacheStorage(config("fp8_e4m3"), mem_pool)
b = AibrixKVCacheStorage(config("fp8_e5m2"), mem_pool)
a.batch_exists(["same_page"])
b.batch_exists(["same_page"])
print("aibrix_e4m3_hash_input=", a.kv_cache_manager.seen[-1])
print("aibrix_e5m2_hash_input=", b.kv_cache_manager.seen[-1])
print("aibrix_dtype_keys_distinct=", a.kv_cache_manager.seen[-1] != b.kv_cache_manager.seen[-1])
assert a.kv_cache_manager.seen[-1] != b.kv_cache_manager.seen[-1], "AIBrix key collision across logical FP8 formats"
