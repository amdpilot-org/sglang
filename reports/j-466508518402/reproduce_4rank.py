"""Exercise the attention CP/TP object broadcast with real Gloo groups."""

import os
import sys
from types import SimpleNamespace

import torch.distributed as dist

from sglang.srt.distributed import communication_op


def main():
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")

    cp_ranks = [rank % 2, rank % 2 + 2]
    tp_ranks = [rank - rank % 2, rank - rank % 2 + 1]

    # Every process creates every group in the same order, as required by torch.
    cp_groups = {
        (0, 2): dist.new_group([0, 2], backend="gloo"),
        (1, 3): dist.new_group([1, 3], backend="gloo"),
    }
    tp_groups = {
        (0, 1): dist.new_group([0, 1], backend="gloo"),
        (2, 3): dist.new_group([2, 3], backend="gloo"),
    }
    cp = SimpleNamespace(
        rank=rank,
        rank_in_group=cp_ranks.index(rank),
        ranks=cp_ranks,
        world_size=2,
        cpu_group=cp_groups[tuple(cp_ranks)],
    )
    tp = SimpleNamespace(
        rank=rank,
        rank_in_group=tp_ranks.index(rank),
        ranks=tp_ranks,
        world_size=2,
        cpu_group=tp_groups[tuple(tp_ranks)],
    )
    communication_op.get_attn_cp_group = lambda: cp
    communication_op.get_attn_tp_group = lambda: tp

    expected = [{"request": 37590}]
    result = communication_op.attn_cp_tp_broadcast_pyobj(
        expected if rank == 0 else None
    )
    if result != expected:
        raise AssertionError(f"rank {rank}: unexpected result {result!r}")

    gathered = [None] * 4 if rank == 0 else None
    dist.gather_object(result, gathered, dst=0)
    if rank == 0:
        print(f"all_ranks={gathered}")
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"rank={os.environ.get('RANK')} exception={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
