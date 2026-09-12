import tempfile
from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS


def _storage_config():
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="logical-anchor-test",
        extra_config={"use_mock_hf3fs_client": True},
    )


def test_hf3fs_factory_uses_marker_page_for_logical_anchor():
    pool = LogicalHostPool(size=8, page_size=4, layout="page_first_direct")

    with patch.object(HiCacheHF3FS, "from_env_config", return_value=object()) as create:
        StorageBackendFactory._create_builtin_backend(
            "hf3fs", HiCacheHF3FS, _storage_config(), pool
        )

    assert create.call_args.args[:2] == (1, torch.uint8)


def test_hf3fs_logical_anchor_marker_round_trip():
    pool = LogicalHostPool(size=8, page_size=4, layout="page_first_direct")
    with tempfile.TemporaryDirectory() as temp_dir:
        backend = HiCacheHF3FS(
            rank=0,
            file_path=f"{temp_dir}/hicache.bin",
            file_size=64,
            numjobs=1,
            bytes_per_page=1,
            entries=2,
            client_timeout=1,
            dtype=torch.uint8,
            metadata_client=Hf3fsLocalMetadataClient(),
            use_mock_client=True,
        )
        try:
            backend.register_mem_pool_host(pool)
            indices = torch.arange(8, dtype=torch.int64)
            assert backend.batch_set_v1(["page-0", "page-1"], indices) == [True, True]
            assert backend.batch_exists(["page-0", "page-1"]) == 2
            assert backend.batch_get_v1(["page-0", "page-1"], indices) == [True, True]
        finally:
            backend.close()


def test_hf3fs_physical_zero_sized_pool_is_not_treated_as_logical():
    class InvalidPhysicalPool:
        layout = "page_first_direct"
        page_size = 4
        dtype = torch.uint8
        kv_buffer = torch.empty(0, dtype=torch.uint8)

        @staticmethod
        def get_ksize_per_token():
            return 0

    with patch.object(
        HiCacheHF3FS, "from_env_config", side_effect=ValueError("zero physical page")
    ) as create:
        with pytest.raises(ValueError, match="zero physical page"):
            StorageBackendFactory._create_builtin_backend(
                "hf3fs", HiCacheHF3FS, _storage_config(), InvalidPhysicalPool()
            )

    assert create.call_args.args[0] == 0


def test_hf3fs_rejects_non_positive_page_size():
    with pytest.raises(ValueError, match="bytes_per_page must be positive"):
        HiCacheHF3FS(
            rank=0,
            file_path="unused",
            file_size=64,
            numjobs=1,
            bytes_per_page=0,
            entries=1,
            client_timeout=1,
            dtype=torch.uint8,
            metadata_client=Hf3fsLocalMetadataClient(),
            use_mock_client=True,
        )
