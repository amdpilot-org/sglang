"""Independent real-Gloo checks for staged attention CP/TP object broadcast."""

import json
import os
from types import SimpleNamespace

import torch.distributed as dist

from sglang.srt.distributed import communication_op


def main():
    rank = int(os.environ["RANK"])
    tp_size = int(os.environ["TEST_TP"])
    cp_size = int(os.environ["TEST_CP"])
    assert int(os.environ["WORLD_SIZE"]) == tp_size * cp_size
    dist.init_process_group("gloo")

    tp_rank = rank % tp_size
    cp_rank = rank // tp_size
    cp_rank_lists = [list(range(t, tp_size * cp_size, tp_size)) for t in range(tp_size)]
    tp_rank_lists = [list(range(c * tp_size, (c + 1) * tp_size)) for c in range(cp_size)]
    cp_groups = {tuple(rs): dist.new_group(rs, backend="gloo") for rs in cp_rank_lists}
    tp_groups = {tuple(rs): dist.new_group(rs, backend="gloo") for rs in tp_rank_lists}
    cp_ranks = cp_rank_lists[tp_rank]
    tp_ranks = tp_rank_lists[cp_rank]
    cp = SimpleNamespace(rank=rank, rank_in_group=cp_rank, ranks=cp_ranks,
                         world_size=cp_size, cpu_group=cp_groups[tuple(cp_ranks)])
    tp = SimpleNamespace(rank=rank, rank_in_group=tp_rank, ranks=tp_ranks,
                         world_size=tp_size, cpu_group=tp_groups[tuple(tp_ranks)])
    communication_op.get_attn_cp_group = lambda: cp
    communication_op.get_attn_tp_group = lambda: tp

    payload = json.loads(os.environ["TEST_PAYLOAD"])
    result = communication_op.attn_cp_tp_broadcast_pyobj(payload if rank == 0 else None)
    assert result == payload, (rank, result, payload)
    gathered = [None] * (tp_size * cp_size) if rank == 0 else None
    dist.gather_object(result, gathered, dst=0)
    if rank == 0:
        print(f"tp={tp_size} cp={cp_size} payload={payload!r} gathered={gathered!r}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
