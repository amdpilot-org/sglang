#!/usr/bin/env python3
import json
import platform
import sys

import torch

from sglang.srt.layers.layernorm import RMSNorm


def main() -> int:
    torch.manual_seed(20260912)
    assert torch.cuda.is_available(), "ROCm GPU is not available through torch.cuda"
    device = torch.device("cuda:0")
    shape = (3, 64)
    dtype = torch.float16
    eps = 1e-6
    atol = 3e-3
    rtol = 3e-3

    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.linspace(0.5, 1.5, shape[-1], device=device, dtype=dtype)
    norm = RMSNorm(shape[-1], eps=eps, weight_dtype=dtype).to(device)
    with torch.no_grad():
        norm.weight.copy_(weight)
        actual = norm(x)
    torch.cuda.synchronize()

    # Independent reference: explicitly compute the RMS normalization in fp32.
    x32 = x.float()
    reference = (
        x32 * torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    ) * weight.float()
    actual32 = actual.float()
    abs_error = (actual32 - reference).abs()
    max_abs_error = abs_error.max().item()
    max_rel_error = (
        abs_error / torch.clamp(reference.abs(), min=torch.finfo(torch.float32).tiny)
    ).max().item()
    passed = torch.allclose(actual32, reference, atol=atol, rtol=rtol)

    result = {
        "result": "pass" if passed else "fail",
        "shape": list(shape),
        "input_dtype": str(dtype),
        "reference_dtype": str(reference.dtype),
        "output_dtype": str(actual.dtype),
        "eps": eps,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "dispatch_method": getattr(norm, "_forward_method").__name__,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count_visible": torch.cuda.device_count(),
        "input": x.cpu().float().tolist(),
        "weight": weight.cpu().float().tolist(),
        "actual": actual32.cpu().tolist(),
        "reference": reference.cpu().tolist(),
        "absolute_error": abs_error.cpu().tolist(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
