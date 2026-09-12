import sys
import types
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig


def cfg(dtype):
    return HiCacheStorageConfig(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
        enable_storage_metrics=False, is_page_first_layout=False,
        model_name="same/model", kv_cache_dtype=dtype,
    )


# Ascend MemCache: exercise the in-tree key transformation without importing
# or contacting the unavailable memcache_hybrid service.
from sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store import NpuMemcacheStore


def npu_keys(dtype):
    obj = NpuMemcacheStore.__new__(NpuMemcacheStore)
    obj.storage_config = cfg(dtype)
    obj.extra_backend_tag = None
    obj._init_runtime_fields(obj.storage_config)
    return obj._tag_keys(["same_page"])


npu_e4 = npu_keys("fp8_e4m3")
npu_e5 = npu_keys("fp8_e5m2")
print("npu_memcache", npu_e4, npu_e5)
assert npu_e4 == npu_e5 == ["same_page"]


# SiMM imports its optional client eagerly. Supply interface-only modules, then
# call the production batch_exists key path on uninitialized service objects.
simm = types.ModuleType("simm")
simm_kv = types.ModuleType("simm.kv")
for name in ("BlockView", "Store"):
    setattr(simm_kv, name, type(name, (), {}))
simm_kv.register_mr = lambda *args: None
simm_kv.set_flag = lambda *args: None
sys.modules["simm"] = simm
sys.modules["simm.kv"] = simm_kv
from sglang.srt.mem_cache.storage.simm.hicache_simm import HiCacheSiMM


def simm_keys(dtype):
    obj = HiCacheSiMM.__new__(HiCacheSiMM)
    obj.storage_config = cfg(dtype)
    obj.is_mla_backend = False
    obj.mha_suffix = "0"
    obj.config = SimpleNamespace(enable_profile=False)
    seen = []
    obj._batch_exist_impl = lambda keys: seen.extend(keys) or [0] * len(keys)
    obj.batch_exists(["same_page"])
    return seen


simm_e4 = simm_keys("fp8_e4m3")
simm_e5 = simm_keys("fp8_e5m2")
print("simm", simm_e4, simm_e5)
assert simm_e4 == simm_e5 == ["same_page_0_k", "same_page_0_v"]
print("REMAINING_COLLISIONS_CONFIRMED")
