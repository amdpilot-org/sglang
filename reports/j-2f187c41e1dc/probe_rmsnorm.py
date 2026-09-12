#!/usr/bin/env python3
"""Reproducible single-GPU numerical probe for SGLang RMSNorm."""

import importlib.util
import json
import os
from pathlib import Path

import torch

from sglang.srt.layers.layernorm import RMSNorm


JOB_ID = "j-2f187c41e1dc"
BASE = "358c163250ad3b1f62939b01ce1314a0a31a0365"
EPS = 1e-6
CASES = ((3, 4096), (3, 4097))
TOLERANCES = {
    torch.float16: {"atol": 2.0e-3, "rtol": 2.0e-3},
    torch.bfloat16: {"atol": 1.6e-2, "rtol": 1.6e-2},
}


def module_origin(name):
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError):
        return None
    return None if spec is None else spec.origin


def main():
    assert torch.cuda.device_count() == 1, "qualification requires exactly one visible GPU"
    torch.cuda.set_device(0)
    torch.manual_seed(20260912)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    records = []

    for dtype in (torch.float16, torch.bfloat16):
        for rows, width in CASES:
            # Generate in float32 so the independent reference uses the exact values
            # subsequently rounded for the implementation input and weight.
            x = (torch.randn(rows, width, device=device, dtype=torch.float32) * 0.7).to(dtype)
            weight = (torch.randn(width, device=device, dtype=torch.float32) * 0.2 + 1.0).to(dtype)
            op = RMSNorm(width, eps=EPS, weight_dtype=dtype).to(device)
            with torch.no_grad():
                op.weight.copy_(weight)
                actual = op(x)
                torch.cuda.synchronize(device)

            x32 = x.float()
            w32 = weight.float()
            reference32 = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + EPS) * w32
            actual32 = actual.float()
            abs_error = (actual32 - reference32).abs()
            rel_error = abs_error / reference32.abs().clamp_min(1e-6)
            tol = TOLERANCES[dtype]
            passed = torch.allclose(actual32, reference32, **tol)
            records.append(
                {
                    "dtype": str(dtype).removeprefix("torch."),
                    "shape": [rows, width],
                    "odd_hidden_width": bool(width % 2),
                    "output_dtype": str(actual.dtype).removeprefix("torch."),
                    "resolved_forward": op._forward_method.__name__,
                    "atol": tol["atol"],
                    "rtol": tol["rtol"],
                    "max_abs_error": abs_error.max().item(),
                    "max_rel_error": rel_error.max().item(),
                    "reference_max_abs": reference32.abs().max().item(),
                    "passed": passed,
                }
            )
            if not passed:
                raise AssertionError(records[-1])

    payload = {
        "job_id": JOB_ID,
        "source_revision": BASE,
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "visible_device_count": torch.cuda.device_count(),
        "visibility_environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        },
        "gpu": {
            "logical_index": 0,
            "name": torch.cuda.get_device_name(device),
            "uuid": str(getattr(props, "uuid", None)),
            "architecture": getattr(props, "gcnArchName", None),
            "total_memory_bytes": props.total_memory,
        },
        "paths": {
            "torch": module_origin("torch"),
            "sglang": module_origin("sglang"),
            "sglang_layernorm": module_origin("sglang.srt.layers.layernorm"),
            "vllm_custom_ops": module_origin("vllm._custom_ops"),
            "vllm_native": module_origin("vllm._C"),
        },
        "reference": "float32 torch: x * rsqrt(mean(x^2) + eps) * weight",
        "epsilon": EPS,
        "measurements": records,
        "all_passed": all(item["passed"] for item in records),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
