import json
import sys

import torch

from sglang.srt.layers.layernorm import RMSNorm


def main() -> int:
    torch.manual_seed(6103717)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    shape = (1, 257)
    eps = 1e-5
    atol = 2e-2
    rtol = 2e-2

    x32_seed = torch.randn(shape, device=device, dtype=torch.float32) * 1.25
    x32_seed[0, 0] = 0.0
    x32_seed[0, 1] = 8.0
    x32_seed[0, -1] = -8.0
    x = x32_seed.to(dtype)
    weight = torch.linspace(0.25, 1.75, shape[-1], device=device, dtype=dtype)
    layer = RMSNorm(shape[-1], eps=eps, weight_dtype=dtype).to(device)
    with torch.no_grad():
        layer.weight.copy_(weight)
        actual = layer(x)
    torch.cuda.synchronize()

    # Independent float32 reference, deliberately avoiding Torch/SGLang RMSNorm APIs.
    x32 = x.float()
    mean_square = (x32 * x32).sum(dim=-1, keepdim=True) / shape[-1]
    reference = x32 * torch.rsqrt(mean_square + eps) * weight.float()
    actual32 = actual.float()
    abs_error = (actual32 - reference).abs()
    passed = torch.allclose(actual32, reference, atol=atol, rtol=rtol)

    props = torch.cuda.get_device_properties(device)
    result = {
        "operation": "sglang.srt.layers.layernorm.RMSNorm GPU forward",
        "boundary": "single-token odd hidden width and bfloat16 dtype",
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
        "absolute_error": abs_error.cpu().tolist(),
        "maximum_absolute_error": abs_error.max().item(),
        "passed": bool(passed),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
