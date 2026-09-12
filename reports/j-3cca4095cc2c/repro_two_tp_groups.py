"""Exercise co-located groups against one synchronized host-memory baseline."""

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import torch
import torch.multiprocessing as mp


def worker(rank, init_file, queue):
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=4
    )
    from sglang.srt.mem_cache.pool_host import base

    gib = 1024**3
    available = 100 * gib
    with (
        mock.patch.object(
            base.psutil,
            "virtual_memory",
            return_value=SimpleNamespace(available=available),
        ),
        mock.patch.object(base, "ranks_per_host", return_value=4),
        mock.patch.object(
            base, "host_memory_sync_group", return_value=torch.distributed.group.WORLD
        ),
    ):
        queue.put((rank, "primary", base.host_memory_budget_bytes(20 * gib)))

    torch.distributed.barrier()
    if rank >= 2:
        # Model a later TP/PP group reaching a sidecar guard after the primary
        # allocations have reduced live host availability. The fixed baseline
        # must be reused without another collective.
        with (
            mock.patch.object(
                base.psutil,
                "virtual_memory",
                return_value=SimpleNamespace(available=60 * gib),
            ),
            mock.patch.object(base, "ranks_per_host", return_value=4),
            mock.patch.object(base.torch.distributed, "all_reduce") as all_reduce,
        ):
            queue.put((rank, "sidecar", base.host_memory_budget_bytes(2 * gib)))
            all_reduce.assert_not_called()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    fd, path = tempfile.mkstemp(prefix="hicache-review-")
    os.close(fd)
    os.unlink(path)
    try:
        mp.spawn(worker, args=(path, queue), nprocs=4)
        values = sorted(queue.get() for _ in range(6))
        printable = [(rank, pool, value / 1024**3) for rank, pool, value in values]
        print(printable)
        primary = [value for _, pool, value in values if pool == "primary"]
        sidecar = [value for _, pool, value in values if pool == "sidecar"]
        assert all(value >= 20 * 1024**3 for value in primary)
        assert all(value >= 2 * 1024**3 for value in sidecar)
    finally:
        if os.path.exists(path):
            os.unlink(path)
