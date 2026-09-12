from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler as sampler_module

rank_values = [[42, 7, 99], [42, 3, 101], [42, 11, 4]]
expected = [min(values) for values in zip(*rank_values)]
token_ids = torch.tensor(rank_values[0], dtype=torch.long, device="cuda:0")
sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
sampler.tp_sync_group = object()


def synthetic_all_gather(output, tensor, group):
    output.copy_(torch.tensor(rank_values, dtype=tensor.dtype, device=tensor.device))


with (
    patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", True),
    patch.object(sampler_module, "is_xpu", return_value=True),
    patch.object(torch.distributed, "get_world_size", return_value=len(rank_values)),
    patch.object(
        torch.distributed,
        "all_gather_into_tensor",
        side_effect=synthetic_all_gather,
    ),
    patch.object(torch.distributed, "all_reduce") as all_reduce,
):
    sampler._sync_token_ids_across_tp(token_ids, SimpleNamespace(grammars=None))

torch.cuda.synchronize()
actual = token_ids.cpu().tolist()
assert actual == expected, (actual, expected)
all_reduce.assert_not_called()
properties = torch.cuda.get_device_properties(0)
print(f"device={properties.name} arch={properties.gcnArchName}")
print(f"rank_values={rank_values}")
print(f"actual={actual} independent_python_min={expected}")
print("limitation=synthetic gather on AMD GPU; no Intel XPU or XCCL execution")
