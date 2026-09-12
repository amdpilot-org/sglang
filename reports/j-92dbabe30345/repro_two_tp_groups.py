import os
import tempfile
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp


def worker(rank, init_file, queue):
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=4
    )
    group_a = torch.distributed.new_group([0, 1], backend="gloo")
    group_b = torch.distributed.new_group([2, 3], backend="gloo")
    from sglang.srt.mem_cache.pool_host import base

    gib = 1024**3
    available = (100 if rank < 2 else 60) * gib
    with (
        __import__("unittest").mock.patch.object(
            base.psutil, "virtual_memory", return_value=SimpleNamespace(available=available)
        ),
        __import__("unittest").mock.patch.object(base, "ranks_per_host", return_value=4),
        __import__("unittest").mock.patch.object(
            base, "host_memory_sync_group", return_value=group_a if rank < 2 else group_b
        ),
    ):
        queue.put((rank, base.host_memory_budget_bytes()))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    fd, path = tempfile.mkstemp(prefix="hicache-review-")
    os.close(fd)
    os.unlink(path)
    try:
        mp.spawn(worker, args=(path, queue), nprocs=4)
        values = sorted(queue.get() for _ in range(4))
        print([(rank, value / 1024**3) for rank, value in values])
        assert all(value >= 20 * 1024**3 for rank, value in values if rank < 2)
        assert all(value < 20 * 1024**3 for rank, value in values if rank >= 2)
    finally:
        if os.path.exists(path):
            os.unlink(path)
