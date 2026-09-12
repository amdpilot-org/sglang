from unittest.mock import Mock, patch

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator


coordinator = GroupCoordinator.__new__(GroupCoordinator)
coordinator.world_size = 2
coordinator.rank_in_group = 0
coordinator.ranks = [0, 1]
coordinator.device_group = object()
coordinator.cpu_group = object()
coordinator.send_object = Mock(return_value=[])

all_gather_group = Mock(world_size=2, rank_in_group=1)
replicated = torch.arange(8.0, device="cuda")
sharded = torch.arange(8.0, device="cuda") + 10

with patch("torch.distributed.send") as send:
    coordinator.send_tensor_dict(
        {"hidden_states": replicated, "sharded": sharded},
        all_gather_group=all_gather_group,
        all_gather_keys={"hidden_states", "residual"},
    )

sent = [call.args[0] for call in send.call_args_list]
assert sent[0].is_cuda and sent[0].numel() == 4
assert sent[1].is_cuda and sent[1].numel() == 8
assert torch.equal(sent[1], sharded)
print(f"device={torch.cuda.get_device_name(0)} arch={torch.cuda.get_device_capability(0)}")
print(f"replicated_sent_numel={sent[0].numel()} sharded_sent_numel={sent[1].numel()}")
