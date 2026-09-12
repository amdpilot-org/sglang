import os

import torch

from sglang.srt.distributed import (
    GroupCoordinator,
)


rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.distributed.init_process_group("gloo")


def make_group(group_ranks, name):
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=0,
        torch_distributed_backend="gloo",
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name=name,
    )


tp_group = make_group([[0, 1], [2, 3]], "repro_tp")
pp_group = make_group([[0, 2], [1, 3]], "repro_pp")
tp_rank = tp_group.rank_in_group
payload = torch.full((1, 8), float(tp_rank + 1), dtype=torch.float32)

if pp_group.is_first_rank:
    pp_group.send_tensor_dict(
        {"sharded": payload, "replicated": torch.arange(8.0)},
        all_gather_group=tp_group,
        all_gather_keys={"replicated"},
    )
else:
    received = pp_group.recv_tensor_dict(
        all_gather_group=tp_group, all_gather_keys={"replicated"}
    )
    expected_sharded = payload
    expected_replicated = torch.arange(8.0)
    print(
        f"rank={rank} tp_rank={tp_rank} "
        f"sharded={received['sharded'].tolist()} "
        f"expected_sharded={expected_sharded.tolist()} "
        f"replicated_ok={torch.equal(received['replicated'], expected_replicated)}",
        flush=True,
    )
    if not torch.equal(received["sharded"], expected_sharded):
        raise AssertionError("TP-sharded tensor was corrupted")
