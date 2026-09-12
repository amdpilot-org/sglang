from types import SimpleNamespace

import torch

import sglang.srt.managers.schedule_batch as schedule_batch

set_mamba_track_indices_from_reqs = schedule_batch.set_mamba_track_indices_from_reqs


device = torch.device("cuda")
batch = SimpleNamespace(
    req_to_token_pool=SimpleNamespace(
        req_index_to_mamba_ping_pong_track_buffer_mapping=torch.tensor(
            [[101, 102], [201, 202]], dtype=torch.int64, device=device
        )
    ),
    req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=device),
    reqs=[
        SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=None)),
        SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=1)),
    ],
)
set_mamba_track_indices_from_reqs(batch)
torch.cuda.synchronize()
actual = batch.mamba_track_indices.cpu().tolist()
print(
    {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "helper": schedule_batch.__file__,
        "actual": actual,
        "safe_expected": [-1, 202],
    }
)
assert actual == [-1, 202]
