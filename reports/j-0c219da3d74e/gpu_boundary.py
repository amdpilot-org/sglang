#!/usr/bin/env python3
"""Independent one-GPU numerical boundary; not a GLM/Mooncake reproduction."""

import json

import torch


torch.manual_seed(32378)
a_cpu = torch.randn(128, 96, dtype=torch.float32)
b_cpu = torch.randn(96, 80, dtype=torch.float32)
reference = a_cpu @ b_cpu
actual = (a_cpu.half().cuda() @ b_cpu.half().cuda()).float().cpu()
result = {
    "device": torch.cuda.get_device_name(0),
    "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "max_abs_error": (actual - reference).abs().max().item(),
    "max_rel_error": ((actual - reference).abs() / reference.abs().clamp_min(1e-5)).max().item(),
    "allclose_rtol_0.01_atol_0.02": torch.allclose(actual, reference, rtol=0.01, atol=0.02),
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["allclose_rtol_0.01_atol_0.02"] else 1)
