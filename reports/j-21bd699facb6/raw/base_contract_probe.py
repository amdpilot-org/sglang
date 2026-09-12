import inspect
import tempfile

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig


def config():
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
        model_name="model",
    )


print("source", inspect.getsourcefile(HiCacheStorageConfig))
print("fields", tuple(HiCacheStorageConfig.__dataclass_fields__))
assert "kv_cache_dtype" not in HiCacheStorageConfig.__dataclass_fields__
with tempfile.TemporaryDirectory() as path:
    bf16_run = HiCacheFile(config(), file_path=path)
    fp8_run = HiCacheFile(config(), file_path=path)
    print("bf16_suffix", bf16_run.config_suffix)
    print("fp8_suffix", fp8_run.config_suffix)
    assert bf16_run._get_suffixed_key("page") == fp8_run._get_suffixed_key("page")
    print("COLLISION_CONFIRMED", bf16_run._get_suffixed_key("page"))
