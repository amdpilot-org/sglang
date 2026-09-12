import inspect
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig


def make_config(dtype=None):
    kwargs = dict(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
        enable_storage_metrics=False, is_page_first_layout=False,
        model_name="same/model",
    )
    if "kv_cache_dtype" in inspect.signature(HiCacheStorageConfig).parameters:
        kwargs["kv_cache_dtype"] = dtype
    return HiCacheStorageConfig(**kwargs)


with tempfile.TemporaryDirectory() as tmp:
    a = HiCacheFile(make_config("bf16"), file_path=tmp)
    b = HiCacheFile(make_config("fp8_e4m3"), file_path=tmp)
    print("file_bf16_suffix=", a.config_suffix)
    print("file_fp8_e4m3_suffix=", b.config_suffix)
    print("file_dtype_keys_distinct=", a._get_suffixed_key("page") != b._get_suffixed_key("page"))

try:
    from sglang.srt.managers import cache_controller
    controller = cache_controller.HiCacheController.__new__(cache_controller.HiCacheController)
    controller.mem_pool_device = SimpleNamespace(dtype=torch.float8_e4m3fnuz)
    controller.mem_pool_host = SimpleNamespace(dtype=torch.uint8, layout="page_first")
    controller.enable_storage_metrics = False
    controller.get_attn_cp_rank_and_size = lambda: (0, 1)
    parallel = SimpleNamespace(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
    vals = []
    for configured in ("fp8_e4m3", "fp8_e5m2"):
        patches = [
            patch.object(cache_controller, "is_dp_attention_enabled", return_value=False),
            patch.object(cache_controller, "get_parallel", return_value=parallel),
        ]
        if hasattr(cache_controller, "get_model"):
            patches.append(patch.object(cache_controller, "get_model", return_value=SimpleNamespace(kv_cache_dtype=configured)))
        for p in patches: p.start()
        try:
            vals.append(getattr(controller._generate_storage_config("same/model"), "kv_cache_dtype", None))
        finally:
            for p in reversed(patches): p.stop()
    print("production_fp8_config_values=", vals)
    print("production_fp8_distinct=", vals[0] != vals[1])
except Exception as exc:
    print("production_config_error=", type(exc).__name__, str(exc))

try:
    from sglang.srt.mem_cache.storage.umbp.umbp_store import UMBPStore
    src = inspect.getsource(UMBPStore.__init__)
    print("umbp_prefix_mentions_dtype=", "kv_cache_dtype" in src)
except Exception as exc:
    print("umbp_inspection_error=", type(exc).__name__, str(exc))
