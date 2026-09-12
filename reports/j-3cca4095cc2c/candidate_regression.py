"""Candidate-only reproduction of the TP-local synchronization counterexample."""

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
    group_a = torch.distributed.new_group([0, 1], backend="gloo")
    group_b = torch.distributed.new_group([2, 3], backend="gloo")
    from sglang.srt.mem_cache.pool_host import base

    available = (100 if rank < 2 else 60) * 1024**3
    with (
        mock.patch.object(
            base.psutil,
            "virtual_memory",
            return_value=SimpleNamespace(available=available),
        ),
        mock.patch.object(base, "ranks_per_host", return_value=4),
        mock.patch.object(
            base,
            "host_memory_sync_group",
            return_value=group_a if rank < 2 else group_b,
        ),
    ):
        queue.put(base.host_memory_budget_bytes())
    torch.distributed.destroy_process_group()


def test_all_co_located_groups_accept_a_fitting_pool():
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    fd, path = tempfile.mkstemp(prefix="hicache-candidate-")
    os.close(fd)
    os.unlink(path)
    try:
        mp.spawn(worker, args=(path, queue), nprocs=4)
        budgets = [queue.get() for _ in range(4)]
        assert all(value >= 20 * 1024**3 for value in budgets), budgets
    finally:
        if os.path.exists(path):
            os.unlink(path)
