#!/usr/bin/env python3
"""Archived copy of the first installed-source SDPA neighboring control."""

import json
import sys
import time

import torch
import torch.nn.functional as F


torch.manual_seed(0)
device = "cuda"
dtype = torch.bfloat16
batch, heads, sequence, head_dim = 2, 8, 512, 128
query = torch.randn((batch, heads, sequence, head_dim), device=device, dtype=dtype)
key = torch.randn_like(query)
value = torch.randn_like(query)
reference = (
    query.float() @ key.float().transpose(-2, -1) * head_dim**-0.5
).softmax(-1) @ value.float()
output = F.scaled_dot_product_attention(query, key, value, is_causal=False)
max_abs_error = (output.float() - reference).abs().max().item()

for _ in range(3):
    F.scaled_dot_product_attention(query, key, value)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(30):
    F.scaled_dot_product_attention(query, key, value)
end.record()
torch.cuda.synchronize()

print(
    json.dumps(
        {
            "python": sys.executable,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "native_torch": torch.__file__,
            "dimensions": {
                "batch": batch,
                "heads": heads,
                "sequence": sequence,
                "head_dim": head_dim,
                "dtype": str(dtype),
            },
            "max_abs_error": max_abs_error,
            "mean_ms_per_forward": start.elapsed_time(end) / 30,
            "max_torch_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "wall_complete_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        indent=2,
    )
)
