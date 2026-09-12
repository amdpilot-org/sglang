#!/usr/bin/env python3
import json
import sys
import torch
from sglang.srt.layers.layernorm import RMSNorm

def main():
    torch.manual_seed(257)
    shape, dtype, eps = (1, 257), torch.bfloat16, 1e-5
    atol, rtol = 0.02, 0.02
    device = torch.device("cuda:0")
    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.linspace(-0.75, 1.25, shape[-1], device=device, dtype=dtype)
    norm = RMSNorm(shape[-1], eps=eps, weight_dtype=dtype).to(device)
    with torch.no_grad():
        norm.weight.copy_(weight)
        actual = norm(x)
    torch.cuda.synchronize()
    x32 = x.float()
    reference = x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + eps) * weight.float()
    actual32 = actual.float()
    error = (actual32 - reference).abs()
    passed = torch.allclose(actual32, reference, atol=atol, rtol=rtol)
    print(json.dumps({
        "result": "pass" if passed else "fail", "shape": list(shape),
        "input_dtype": str(dtype), "output_dtype": str(actual.dtype),
        "reference_dtype": str(reference.dtype), "eps": eps, "atol": atol,
        "rtol": rtol, "max_abs_error": error.max().item(),
        "max_rel_error": (error / reference.abs().clamp_min(torch.finfo(torch.float32).tiny)).max().item(),
        "dispatch_method": norm._forward_method.__name__, "torch_version": torch.__version__,
        "hip_version": torch.version.hip, "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count_visible": torch.cuda.device_count(), "input": x.float().cpu().tolist(),
        "weight": weight.float().cpu().tolist(), "actual": actual32.cpu().tolist(),
        "reference": reference.cpu().tolist(), "absolute_error": error.cpu().tolist()
    }, indent=2, sort_keys=True))
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())
