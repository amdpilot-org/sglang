import json
import sys

import torch

from sglang.srt.layers.layernorm import RMSNorm


def main() -> int:
    torch.manual_seed(6103716)
    device = torch.device("cuda:0")
    dtype = torch.float16
    shape = (3, 64)
    eps = 1e-6
    atol = 3e-3
    rtol = 3e-3

    x = (torch.randn(shape, device=device, dtype=torch.float32) * 0.75).to(dtype)
    weight = torch.linspace(0.5, 1.5, shape[-1], device=device, dtype=dtype)
    layer = RMSNorm(shape[-1], eps=eps, weight_dtype=dtype).to(device)
    with torch.no_grad():
        layer.weight.copy_(weight)
        actual = layer(x)
    torch.cuda.synchronize()

    # Independent reference: explicitly perform the RMS reduction and scaling
    # in float32, without torch.nn.functional.rms_norm or SGLang kernels.
    x32 = x.float()
    reference = x32 * torch.rsqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps)
    reference = reference * weight.float()
    actual32 = actual.float()
    abs_error = (actual32 - reference).abs()
    passed = torch.allclose(actual32, reference, atol=atol, rtol=rtol)

    props = torch.cuda.get_device_properties(device)
    result = {
        "operation": "sglang.srt.layers.layernorm.RMSNorm GPU forward",
        "device": {
            "index": 0,
            "name": torch.cuda.get_device_name(device),
            "architecture": getattr(props, "gcnArchName", None),
            "total_memory_bytes": props.total_memory,
        },
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "shape": list(shape),
        "dtype": str(dtype),
        "epsilon": eps,
        "tolerance": {"atol": atol, "rtol": rtol},
        "input": x.cpu().tolist(),
        "weight": weight.cpu().tolist(),
        "actual": actual.cpu().tolist(),
        "float32_reference": reference.cpu().tolist(),
        "maximum_absolute_error": abs_error.max().item(),
        "passed": bool(passed),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
