from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)


def _controller(*, should_backup):
    controller = object.__new__(HybridCacheController)
    controller.backup_skip = True
    controller.page_size = 64
    controller.should_backup = should_backup
    controller.storage_backend = Mock()
    controller._resolve_sidecar_kv_derived_pool_transfers = Mock()
    controller._resolve_sidecar_nonkv_derived_pool_transfers = Mock()
    return controller


def _operation(pool_transfers):
    return SimpleNamespace(
        pool_transfers=pool_transfers,
        hash_value=["page-0", "page-1"],
        completed_tokens=None,
        pool_storage_result=Mock(),
    )


def test_backup_skip_without_rank_local_sidecars_is_successful():
    """TP followers rely on TP0 for replicated MLA/DSV4 pools."""
    controller = _controller(should_backup=lambda transfer: False)
    operation = _operation([])

    controller._page_backup(operation)

    assert operation.completed_tokens == 128
    controller.storage_backend.batch_set_v2.assert_not_called()


def test_backup_skip_with_successful_rank_local_sidecar_is_successful():
    transfer = SimpleNamespace(
        name=PoolName.MAMBA,
        keys=["page-0", "page-1"],
        host_indices=None,
    )
    controller = _controller(should_backup=lambda candidate: candidate is transfer)
    controller.storage_backend.batch_set_v2.return_value = {
        PoolName.MAMBA: [True, True]
    }
    operation = _operation([transfer])

    controller._page_backup(operation)

    assert operation.completed_tokens == 128


def test_backup_skip_with_failed_rank_local_sidecar_is_unsuccessful():
    transfer = SimpleNamespace(
        name=PoolName.MAMBA,
        keys=["page-0", "page-1"],
        host_indices=None,
    )
    controller = _controller(should_backup=lambda candidate: candidate is transfer)
    controller.storage_backend.batch_set_v2.return_value = {
        PoolName.MAMBA: [True, False]
    }
    operation = _operation([transfer])

    controller._page_backup(operation)

    assert operation.completed_tokens == 0
