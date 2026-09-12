import os
import tempfile
from multiprocessing import get_context
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.srt.layers.sampler import Sampler


def worker(rank, init_file, queue):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        sampler = Sampler.__new__(Sampler)
        sampler.tp_sync_group = dist.group.WORLD
        sampler.tp_grammar_entry_only = True
        token = torch.tensor([3 if rank == 0 else 9], dtype=torch.int64)
        sampler._sync_token_ids_across_tp(token, SimpleNamespace(grammars=[object()]))
        queue.put((rank, token.item()))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ctx = get_context("spawn")
    queue = ctx.Queue()
    fd, init_file = tempfile.mkstemp(prefix="grammar-review-", dir=os.environ["REVIEW_RUNTIME"])
    os.close(fd)
    os.unlink(init_file)
    procs = [ctx.Process(target=worker, args=(rank, init_file, queue)) for rank in range(2)]
    for proc in procs:
        proc.start()
    results = sorted(queue.get(timeout=60) for _ in procs)
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0, proc.exitcode
    print(results)
    assert results == [(0, 3), (1, 3)]
