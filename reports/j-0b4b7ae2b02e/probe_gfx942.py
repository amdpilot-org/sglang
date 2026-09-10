import argparse
import importlib
import json
import platform

import torch


DEFAULT_MODULE = "sglang.kernels.ops.diffusion.norm.fused_residual_norm_flydsl"


def reference(residual, x, gate, weight, bias, scale, shift, norm_type):
    def as_bfloat16(value):
        return value.to(torch.bfloat16)

    residual_bf16 = as_bfloat16(residual)
    x_bf16 = as_bfloat16(x)
    gate_bf16 = as_bfloat16(gate)
    weight_bf16 = as_bfloat16(weight)
    bias_bf16 = as_bfloat16(bias)
    scale_bf16 = as_bfloat16(scale)
    shift_bf16 = as_bfloat16(shift)

    residual_sum = residual_bf16.float() + x_bf16.float() * gate_bf16.float()
    residual_out = residual_sum.to(torch.bfloat16)
    base = residual_out.float()
    if norm_type == "layer":
        mean = base.mean(dim=-1, keepdim=True)
        variance = base.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (base - mean) * torch.rsqrt(variance + 1e-6)
    else:
        variance = base.pow(2).mean(dim=-1, keepdim=True)
        normalized = base * torch.rsqrt(variance + 1e-6)
    normalized = normalized * weight_bf16.float() + bias_bf16.float()
    output = (
        normalized * (1.0 + scale_bf16.float()) + shift_bf16.float()
    ).to(torch.bfloat16)
    return output, residual_out


def error_metrics(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": difference.max().item(),
        "max_rel": (difference / denominator).max().item(),
    }


def run_supported_case(op, dtype, dimension, norm_type):
    torch.manual_seed(42)
    shape = (1, 1, dimension)
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    gate = torch.randn((1, 1, dimension), device="cuda", dtype=dtype)
    weight = torch.randn((dimension,), device="cuda", dtype=dtype)
    bias = torch.randn((dimension,), device="cuda", dtype=dtype)
    scale = torch.randn((1, 1, dimension), device="cuda", dtype=dtype)
    shift = torch.randn((1, 1, dimension), device="cuda", dtype=dtype)

    output, residual_out = op(
        residual, x, gate, weight, bias, scale, shift, norm_type, 1e-6
    )
    expected_output, expected_residual = reference(
        residual, x, gate, weight, bias, scale, shift, norm_type
    )
    torch.testing.assert_close(residual_out, expected_residual, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(output, expected_output, atol=1.0, rtol=5e-2)
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "dimension": dimension,
        "norm_type": norm_type,
        "output_dtype": str(output.dtype).removeprefix("torch."),
        "residual_error": error_metrics(residual_out, expected_residual),
        "output_error": error_metrics(output, expected_output),
        "status": "PASS",
    }


def run_tail_case(op, dimension):
    torch.manual_seed(42)
    shape = (1, 1, dimension)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((dimension,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((dimension,), device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    try:
        op(
            residual, x, gate, weight, bias, scale, shift, "rms", 1e-6
        )
    except AssertionError as exc:
        return {
            "dimension": dimension,
            "status": "REJECTED",
            "exception": f"{type(exc).__name__}: {exc}",
        }
    return {"dimension": dimension, "status": "UNEXPECTEDLY_ACCEPTED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    op = module.flydsl_fused_residual_norm_scale_shift
    device = torch.cuda.get_device_properties(0)
    results = {
        "module": args.module,
        "module_path": module.__file__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": {
            "name": device.name,
            "gcn_arch_name": device.gcnArchName,
            "compute_capability": f"{device.major}.{device.minor}",
        },
        "supported": [
            run_supported_case(op, torch.bfloat16, 5120, "rms"),
            run_supported_case(op, torch.bfloat16, 5120, "layer"),
            run_supported_case(op, torch.float16, 5120, "rms"),
            run_supported_case(op, torch.float32, 5120, "rms"),
            run_supported_case(op, torch.bfloat16, 10240, "rms"),
        ],
        "tails": [
            run_tail_case(op, dimension)
            for dimension in (5119, 5121, 10239)
        ],
    }
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")
        print(rendered)


if __name__ == "__main__":
    main()
