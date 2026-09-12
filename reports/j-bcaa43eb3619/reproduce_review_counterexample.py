import threading
from queue import Queue
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import HiCacheController, StorageOperation
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)


def run(worker_cls, controller, exceptions):
    controller.storage_stop_event = threading.Event()
    controller.backup_skip = False
    controller.page_size = 16
    controller.backup_queue = Queue()
    controller.ack_backup_queue = Queue()
    failures = iter(exceptions)

    def page_backup(operation):
        try:
            raise next(failures)
        except StopIteration:
            operation.completed_tokens = 16 * len(operation.hash_value)

    controller._page_backup = page_backup
    operations = [
        StorageOperation(torch.zeros(16), list(range(16)), hash_value=[f"h{i}"])
        for i in range(3)
    ]
    for operation in operations:
        controller.backup_queue.put(operation)
    thread = threading.Thread(
        target=worker_cls.backup_thread_func, args=(controller,), daemon=True
    )
    thread.start()
    acks = [controller.ack_backup_queue.get(timeout=5) for _ in operations]
    controller.storage_stop_event.set()
    controller.backup_queue.put(None)
    thread.join(timeout=5)
    fields = [
        {
            "failed": ack.failed,
            "failure_kind": ack.failure_kind,
            "unwritten_pages": ack.unwritten_pages,
            "exception_type": getattr(ack, "failure_exception_type", None),
            "exception_message": getattr(ack, "failure_exception_message", None),
        }
        for ack in acks
    ]
    print(worker_cls.__name__, fields)
    return fields[0] != fields[1]


results = []
for cls in (HiCacheController, HybridCacheController):
    controller = mock.MagicMock() if cls is HiCacheController else cls.__new__(cls)
    results.append(
        run(
            cls,
            controller,
            [TimeoutError("remote timed out"), ValueError("malformed response")],
        )
    )
assert all(results), "distinct exceptions have indistinguishable ack metadata"
