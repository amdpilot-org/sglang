#!/usr/bin/env python3

import argparse
import inspect
import json
import os
import platform
from pathlib import Path

import torch


def independent_gemma_rmsnorm(input_tensor, weight, eps):
    input_float = input_tensor.float()
    mean_square = input_float.pow(2).mean(-1, keepdim=True)
    normalized = input_float * torch.rsqrt(mean_square + eps)
    return (normalized * (1.0 + weight.float())).type_as(input_tensor)


def comparison_metrics(actual, expected):
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "allclose_rtol_1e_2_atol_1e_2": bool(
            torch.allclose(actual_float, expected_float, rtol=1e-2, atol=1e-2)
        ),
        "nonfinite": bool(torch.isnan(actual).any() or torch.isinf(actual).any()),
    }


def make_layer(dim, device):
    from sglang.srt.layers.layernorm import Gemma3RMSNorm

    layer = Gemma3RMSNorm(dim, eps=1e-6).to(device)
    layer.weight.data.normal_(mean=0.0, std=0.1)
    return layer


def make_input(shape, dtype, device, layout="contiguous"):
    scale = 1.0 / (2 * shape[-1])
    if layout == "transposed":
        base = torch.randn(shape[1], shape[0], shape[2], device=device) * scale
        return base.transpose(0, 1).to(dtype)
    if layout == "sliced":
        wide = torch.randn(*shape[:-1], shape[-1] * 2, device=device) * scale
        return wide[..., : shape[-1]].to(dtype)
    return (torch.randn(*shape, device=device) * scale).to(dtype)


def timed_call(callable_object, warmup_count=5, iteration_count=50):
    for _ in range(warmup_count):
        callable_object()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iteration_count):
        callable_object()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iteration_count * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("main", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import sgl_kernel
    import sglang
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
        gemma_fused_add_rmsnorm as rocm_triton_fused_add,
    )
    from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
        gemma_rmsnorm as rocm_triton_rmsnorm,
    )
    from sglang.srt.layers import layernorm
    from sglang.srt.layers.layernorm import Gemma3RMSNorm

    kernel_calls = {"rmsnorm": 0, "fused_add_rmsnorm": 0}

    def counted_rmsnorm(input_tensor, weight, eps):
        kernel_calls["rmsnorm"] += 1
        return rocm_triton_rmsnorm(input_tensor, weight, eps)

    def counted_fused_add(input_tensor, residual, weight, eps):
        kernel_calls["fused_add_rmsnorm"] += 1
        output, residual_output = rocm_triton_fused_add(
            input_tensor, residual, weight, eps
        )
        input_tensor.copy_(output)
        residual.copy_(residual_output)
        return input_tensor, residual

    if args.mode == "candidate":
        layernorm.gemma_rmsnorm = counted_rmsnorm
        layernorm.gemma_fused_add_rmsnorm = counted_fused_add

    torch.manual_seed(32807)
    device = "cuda"
    gpu_properties = torch.cuda.get_device_properties(0)
    dim = 256
    tokens = 37
    heads = 4
    head_dim = 64
    layer = make_layer(dim, device)
    results = {}

    for activation_dtype in (torch.bfloat16, torch.float16):
        for weight_dtype in (activation_dtype, torch.float32):
            for case_name, shape, layout in (
                ("hidden_2d", (64, dim), "contiguous"),
                ("qk_3d_contiguous", (tokens, heads, head_dim), "contiguous"),
                ("qk_3d_transposed", (tokens, heads, head_dim), "transposed"),
                ("qk_3d_sliced", (tokens, heads, head_dim), "sliced"),
            ):
                layer = make_layer(shape[-1], device)
                layer.weight.data = layer.weight.data.to(weight_dtype).contiguous()
                input_tensor = make_input(shape, activation_dtype, device, layout)
                expected = independent_gemma_rmsnorm(
                    input_tensor, layer.weight.data, layer.eps
                )
                calls_before = dict(kernel_calls)
                if args.mode == "candidate":
                    actual = layer.forward_cuda(input_tensor)
                else:
                    actual = layer(input_tensor)
                torch.cuda.synchronize()
                module_case = {
                    "dispatch": (
                        "forward_cuda" if args.mode == "candidate" else "module_forward"
                    ),
                    "comparison": comparison_metrics(actual, expected),
                    "kernel_calls": {
                        name: count - calls_before[name]
                        for name, count in kernel_calls.items()
                    },
                }
                direct_actual = rocm_triton_rmsnorm(
                    input_tensor, layer.weight.data, layer.eps
                )
                torch.cuda.synchronize()
                results[
                    f"{case_name}_{str(activation_dtype).split('.')[-1]}"
                    f"_weight_{str(weight_dtype).split('.')[-1]}"
                ] = {
                    "input_contiguous": bool(input_tensor.is_contiguous()),
                    "module": module_case,
                    "rocm_triton_direct": comparison_metrics(direct_actual, expected),
                }

    residual_input = make_input((83, dim), torch.bfloat16, device)
    residual_tensor = make_input((83, dim), torch.bfloat16, device)
    layer = make_layer(dim, device)
    layer.weight.data = layer.weight.data.to(torch.bfloat16).contiguous()
    residual_reference = residual_input + residual_tensor
    residual_expected = independent_gemma_rmsnorm(
        residual_reference, layer.weight.data, layer.eps
    )
    calls_before = dict(kernel_calls)
    if args.mode == "candidate":
        residual_output, updated_residual = layer.forward_cuda(
            residual_input.clone(), residual_tensor.clone()
        )
    else:
        residual_output, updated_residual = layer(
            residual_input.clone(), residual_tensor.clone()
        )
    torch.cuda.synchronize()
    results["residual_bf16_bf16_weight"] = {
        "module": {
            "dispatch": (
                "forward_cuda" if args.mode == "candidate" else "module_forward"
            ),
            "output": comparison_metrics(residual_output, residual_expected),
            "residual": comparison_metrics(updated_residual, residual_reference),
            "kernel_calls": {
                name: count - calls_before[name] for name, count in kernel_calls.items()
            },
        }
    }

    timing_us = {}
    if args.mode == "main":
        timing_input = make_input((64, dim), torch.bfloat16, device)
        timing_input_3d = make_input(
            (tokens, heads, head_dim), torch.bfloat16, device
        )
        timing_layer_3d = make_layer(head_dim, device)
        timing_layer_3d.weight.data = (
            timing_layer_3d.weight.data.to(torch.bfloat16).contiguous()
        )
        timing_us["module_forward_2d_bf16"] = timed_call(
            lambda: layer(timing_input)
        )
        timing_us["rocm_triton_direct_2d_bf16"] = timed_call(
            lambda: rocm_triton_rmsnorm(timing_input, layer.weight.data, layer.eps)
        )
        timing_us["module_forward_3d_bf16"] = timed_call(
            lambda: timing_layer_3d(timing_input_3d)
        )
        timing_us["rocm_triton_direct_3d_bf16"] = timed_call(
            lambda: rocm_triton_rmsnorm(
                timing_input_3d, timing_layer_3d.weight.data, timing_layer_3d.eps
            )
        )

    dispatch_method = (
        layer.dispatch_forward() if args.mode == "main" else layer.forward_cuda
    )
    record = {
        "mode": args.mode,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "total_memory_bytes": int(gpu_properties.total_memory),
        },
        "environment": {
            "python": os.path.abspath(__import__("sys").executable),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_hip_version": str(torch.version.hip),
            "torch_path": os.path.dirname(torch.__file__),
            "sglang_version": getattr(sglang, "__version__", None),
            "sglang_path": os.path.abspath(sglang.__file__),
            "sgl_kernel_version": getattr(sgl_kernel, "__version__", None),
            "sgl_kernel_path": os.path.abspath(sgl_kernel.__file__),
            "layernorm_path": inspect.getsourcefile(Gemma3RMSNorm),
            "rocm_triton_path": inspect.getsourcefile(rocm_triton_rmsnorm),
        },
        "dispatch_method": dispatch_method.__name__,
        "timing_method": (
            "CUDA events, 5 warmups + 50 timed iterations, one call per iteration"
        ),
        "timing_us_per_call": timing_us,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "dispatch_method": record["dispatch_method"],
                "timing_us_per_call": timing_us,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
