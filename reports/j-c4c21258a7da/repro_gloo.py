import os
import tempfile

import torch
import torch.multiprocessing as mp


def worker(rank, world_size, init_file, queue):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    from sglang.srt.mem_cache.pool_host import base

    gib = 1024**3
    reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
    readings = [reserve + 64 * gib, reserve + 40 * gib, reserve + 52 * gib]
    original_virtual_memory = base.psutil.virtual_memory
    original_ranks_per_host = base.ranks_per_host
    original_sync_group = base.host_memory_sync_group
    try:
        base.psutil.virtual_memory = lambda: type(
            "Memory", (), {"available": readings[rank]}
        )()
        base.ranks_per_host = lambda: world_size
        base.host_memory_sync_group = lambda: torch.distributed.group.WORLD
        queue.put((rank, base.host_memory_budget_bytes()))
    finally:
        base.psutil.virtual_memory = original_virtual_memory
        base.ranks_per_host = original_ranks_per_host
        base.host_memory_sync_group = original_sync_group
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    world_size = 3
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    fd, init_file = tempfile.mkstemp(prefix="hicache-gloo-")
    os.close(fd)
    os.unlink(init_file)
    try:
        mp.spawn(worker, args=(world_size, init_file, queue), nprocs=world_size)
        results = sorted(queue.get() for _ in range(world_size))
        expected = 40 * (1024**3) // world_size
        print({"results": results, "expected_each": expected})
        assert all(budget == expected for _, budget in results)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
