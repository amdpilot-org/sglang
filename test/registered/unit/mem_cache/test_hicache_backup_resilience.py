import threading
from queue import Queue
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import HiCacheController, StorageOperation
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    StorageOperation as HybridStorageOperation,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_op(num_pages, page_size=16):
    return StorageOperation(
        torch.zeros(num_pages * page_size),
        list(range(num_pages * page_size)),
        hash_value=[f"h{i}" for i in range(num_pages)],
    )


def run_worker(worker_cls, controller):
    thread = threading.Thread(
        target=worker_cls.backup_thread_func, args=(controller,), daemon=True
    )
    thread.start()
    return thread


def stop_worker(controller, thread):
    controller.storage_stop_event.set()
    controller.backup_queue.put(None)
    thread.join(timeout=5)
    assert not thread.is_alive()


def make_controller(page_backup, backup_skip=False):
    controller = mock.MagicMock()
    controller.storage_stop_event = threading.Event()
    controller.backup_skip = backup_skip
    controller.page_size = 16
    controller.backup_queue = Queue()
    controller.ack_backup_queue = Queue()
    controller._page_backup = page_backup
    return controller


def test_worker_survives_exception_and_drains_following_operation():
    calls = 0

    def fail_once(operation):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("storage backend I/O error")
        operation.completed_tokens = 16 * len(operation.hash_value)

    controller = make_controller(fail_once)
    first = make_op(4)
    second = make_op(2)
    controller.backup_queue.put(first)
    controller.backup_queue.put(second)

    thread = run_worker(HiCacheController, controller)
    ack1 = controller.ack_backup_queue.get(timeout=5)
    ack2 = controller.ack_backup_queue.get(timeout=5)
    assert thread.is_alive()
    stop_worker(controller, thread)

    assert ack1 is first
    assert ack1.failed
    assert ack1.failure_kind == "exception"
    assert ack1.failure_exception_type == "RuntimeError"
    assert ack1.failure_exception_message == "storage backend I/O error"
    assert ack1.unwritten_pages == 4
    assert ack2 is second
    assert not ack2.failed
    assert ack2.failure_exception_type is None
    assert ack2.failure_exception_message is None
    assert ack2.completed_tokens == 32
    assert controller.backup_queue.empty()


def test_exception_after_partial_progress_counts_only_remaining_pages():
    def partial_then_fail(operation):
        operation.completed_tokens = 48
        raise RuntimeError("third batch failed")

    controller = make_controller(partial_then_fail)
    operation = make_op(4)
    controller.backup_queue.put(operation)
    thread = run_worker(HiCacheController, controller)
    ack = controller.ack_backup_queue.get(timeout=5)
    stop_worker(controller, thread)

    assert ack.failed
    assert ack.unwritten_pages == 1


def test_exception_ack_preserves_distinct_causes():
    failures = iter(
        [TimeoutError("remote timed out"), ValueError("malformed response")]
    )

    def fail(operation):
        raise next(failures)

    controller = make_controller(fail)
    controller.backup_queue.put(make_op(1))
    controller.backup_queue.put(make_op(1))
    thread = run_worker(HiCacheController, controller)
    timeout_ack = controller.ack_backup_queue.get(timeout=5)
    malformed_ack = controller.ack_backup_queue.get(timeout=5)
    stop_worker(controller, thread)

    assert timeout_ack.failure_exception_type == "TimeoutError"
    assert timeout_ack.failure_exception_message == "remote timed out"
    assert malformed_ack.failure_exception_type == "ValueError"
    assert malformed_ack.failure_exception_message == "malformed response"


def test_backend_false_marks_all_unwritten_pages_after_completed_batches():
    controller = mock.MagicMock()
    controller.page_size = 16
    controller.page_set_func = mock.MagicMock(side_effect=[True, False])
    operation = make_op(129)

    HiCacheController._page_backup(controller, operation)

    assert operation.failed
    assert operation.failure_kind == "backend_false"
    assert operation.completed_tokens == 128 * 16
    assert operation.unwritten_pages == 1


def test_success_and_backup_skip_leave_operation_clean():
    direct = mock.MagicMock()
    direct.page_size = 16
    direct.page_set_func = mock.MagicMock(return_value=True)
    successful = make_op(2)
    HiCacheController._page_backup(direct, successful)
    assert not successful.failed
    assert successful.completed_tokens == 32

    skipped_call = mock.MagicMock(side_effect=AssertionError("must not run"))
    controller = make_controller(skipped_call, backup_skip=True)
    skipped = make_op(2)
    controller.backup_queue.put(skipped)
    thread = run_worker(HiCacheController, controller)
    ack = controller.ack_backup_queue.get(timeout=5)
    stop_worker(controller, thread)
    assert ack is skipped
    assert not ack.failed
    skipped_call.assert_not_called()


def test_hybrid_worker_survives_exception_and_acks_operation():
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.storage_stop_event = threading.Event()
    controller.backup_skip = False
    controller.page_size = 16
    controller.backup_queue = Queue()
    controller.ack_backup_queue = Queue()
    controller._page_backup = mock.MagicMock(
        side_effect=RuntimeError("sidecar backend I/O error")
    )
    operation = HybridStorageOperation(
        torch.zeros(32), list(range(32)), hash_value=["h0", "h1"]
    )
    controller.backup_queue.put(operation)

    thread = run_worker(HybridCacheController, controller)
    ack = controller.ack_backup_queue.get(timeout=5)
    stop_worker(controller, thread)

    assert ack is operation
    assert ack.failed
    assert ack.failure_kind == "exception"
    assert ack.failure_exception_type == "RuntimeError"
    assert ack.failure_exception_message == "sidecar backend I/O error"
    assert ack.unwritten_pages == 2


def test_hybrid_partial_sidecar_failure_is_reported():
    transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.zeros(3),
        keys=["i0", "i1", "i2"],
    )
    operation = HybridStorageOperation(
        torch.zeros(48),
        list(range(48)),
        hash_value=["h0", "h1", "h2"],
        pool_transfers=[transfer],
    )
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = 16
    controller.backup_skip = False
    controller.storage_backend_type = "test"
    controller.storage_backend = mock.MagicMock()
    controller.storage_backend.batch_set_v2.return_value = {
        PoolName.INDEXER: [True, False, False]
    }
    controller._resolve_sidecar_kv_derived_pool_transfers = lambda op: None
    controller._resolve_sidecar_nonkv_derived_pool_transfers = lambda op: None

    with mock.patch.object(HiCacheController, "_page_backup", return_value=None):
        HybridCacheController._page_backup(controller, operation)

    assert operation.failed
    assert operation.failure_kind == "sidecar_backend_false"
    assert operation.sidecar_unwritten_by_pool == {PoolName.INDEXER: 2}
