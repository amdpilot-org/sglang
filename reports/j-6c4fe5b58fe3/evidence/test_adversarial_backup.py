from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController


def controller(should_backup):
    value = object.__new__(HybridCacheController)
    value.backup_skip = True
    value.page_size = 64
    value.should_backup = should_backup
    value.storage_backend = Mock()
    value._resolve_sidecar_kv_derived_pool_transfers = Mock()
    value._resolve_sidecar_nonkv_derived_pool_transfers = Mock()
    return value


def operation(transfers):
    return SimpleNamespace(
        pool_transfers=transfers,
        hash_value=["p0", "p1"],
        completed_tokens=None,
        pool_storage_result=Mock(),
    )


@pytest.mark.parametrize("transfers", [None, []])
def test_no_local_work_is_success(transfers):
    c = controller(lambda _: False)
    op = operation(transfers)
    c._page_backup(op)
    assert op.completed_tokens == 128
    c.storage_backend.batch_set_v2.assert_not_called()


def test_filtered_replicated_transfer_is_success_without_write():
    transfer = SimpleNamespace(name=PoolName.SWA, keys=["p1"], host_indices=None)
    c = controller(lambda _: False)
    op = operation([transfer])
    c._page_backup(op)
    assert op.completed_tokens == 128
    c.storage_backend.batch_set_v2.assert_not_called()


@pytest.mark.parametrize("result", [{}, {PoolName.MAMBA: [True]}, {PoolName.MAMBA: [True, False]}])
def test_selected_transfer_requires_complete_success(result):
    transfer = SimpleNamespace(
        name=PoolName.MAMBA, keys=["p0", "p1"], host_indices=None
    )
    c = controller(lambda _: True)
    c.storage_backend.batch_set_v2.return_value = result
    op = operation([transfer])
    c._page_backup(op)
    assert op.completed_tokens == 0


def test_string_key_backend_result_is_accepted():
    transfer = SimpleNamespace(
        name=PoolName.MAMBA, keys=None, host_indices=torch.tensor([4, 5])
    )
    c = controller(lambda _: True)
    c.storage_backend.batch_set_v2.return_value = {PoolName.MAMBA.value: [True, True]}
    op = operation([transfer])
    c._page_backup(op)
    assert op.completed_tokens == 128
