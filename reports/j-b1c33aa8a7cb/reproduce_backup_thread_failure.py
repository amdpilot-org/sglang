import threading
from queue import Empty, Queue
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import HiCacheController, StorageOperation

controller = mock.MagicMock()
controller.storage_stop_event = threading.Event()
controller.backup_skip = False
controller.page_size = 16
controller.backup_queue = Queue()
controller.ack_backup_queue = Queue()

calls = [0]


def page_backup(operation):
    calls[0] += 1
    if calls[0] == 1:
        raise RuntimeError("storage backend I/O error")
    operation.completed_tokens = 16 * len(operation.hash_value)


controller._page_backup = page_backup


def make_op(n):
    return StorageOperation(
        torch.zeros(n * 16),
        list(range(n * 16)),
        hash_value=[f"h{i}" for i in range(n)],
    )


controller.backup_queue.put(make_op(4))
controller.backup_queue.put(make_op(2))

t = threading.Thread(
    target=HiCacheController.backup_thread_func, args=(controller,), daemon=True
)
t.start()
t.join(timeout=3)
print("backup thread alive:", t.is_alive())
acked = []
try:
    while True:
        acked.append(controller.ack_backup_queue.get(timeout=1).id)
except Empty:
    pass
print("acked operations:", acked, "(2 expected)")
print("still queued:", controller.backup_queue.qsize())

assert t.is_alive(), "backup worker died after a backend exception"
assert len(acked) == 2, "both operations must be acked"
assert controller.backup_queue.empty(), "backup queue must keep draining"
