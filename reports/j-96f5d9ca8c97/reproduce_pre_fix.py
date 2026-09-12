"""Reproduce the tensor construction failure fixed by upstream PR #27998."""

from types import SimpleNamespace

import torch


reqs = [SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=None))]
print(
    f"device={torch.cuda.get_device_name(0)} "
    f"arch={torch.cuda.get_device_properties(0).gcnArchName}"
)

try:
    torch.tensor(
        [req.kv.mamba_next_track_idx for req in reqs],
        dtype=torch.int64,
        pin_memory=True,
    )
except TypeError as exc:
    print(f"reproduced={type(exc).__name__}: {exc}")
else:
    raise AssertionError("pre-fix expression unexpectedly accepted None")
