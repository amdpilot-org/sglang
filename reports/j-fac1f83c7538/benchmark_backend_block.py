#!/usr/bin/env python3
"""Bounded MI300X comparison of SGLang's public reduced transformer block."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Callable


CASES = ((4, 256, 128), (8, 256, 128), (16, 256, 128))
WARMUP_ITERATIONS = 5
TIMED_ITERATIONS = 20
SEED = 20260910
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("aiter", "torch"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cuda_timing(function: Callable[[], object]) -> dict[str, float]:
    for _ in range(WARMUP_ITERATIONS):
        function()
    torch.cuda.synchronize()

    samples = []
    for _ in range(TIMED_ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p95_ms": statistics.quantiles(samples, n=20)[-1],
        "samples_ms": samples,
    }


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "max_rel": (delta / denominator).max().item(),
    }


def loaded_native_paths() -> list[str]:
    paths = set()
    patterns = ("libtorch", "libhipblas", "librocblas", "module_aiter_core")
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            path = line.rstrip().split(" ", maxsplit=5)[-1]
            if any(pattern in path for pattern in patterns) and os.path.isfile(path):
                paths.add(path)
    return sorted(paths)


def source_path(obj: object) -> str | None:
    try:
        return inspect.getsourcefile(obj)
    except (TypeError, OSError):
        return None


def main() -> None:
    args = parse_args()
    os.environ["SGLANG_FORCE_FUSED_OP_BACKEND"] = args.backend
    os.environ["SGLANG_USE_AITER"] = "1" if args.backend == "aiter" else "0"

    global torch
    import torch
    import sglang
    from aiter import rmsnorm2d_fwd, silu_and_mul
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.tuned_gemm import get_GEMM_A16W16_config
    from aiter.tuned_gemm import tgemm
    from sglang.kernels.fused_op import (
        clear_fused_op_trace,
        disable_fused_op_trace,
        enable_fused_op_trace,
        get_fused_op_backend,
        get_fused_op_trace,
    )
    from sglang.kernels.ops.activation import silu_and_mul as sglang_silu_and_mul
    from sglang.kernels.ops.attention.dsv4 import linear_bf16_fp32
    from sglang.kernels.ops.attention.dsv4 import gemm as gemm_module
    from sglang.kernels.ops.layernorm import rmsnorm

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    results = []
    gfx = get_gfx()
    cu_num = get_cu_num()

    enable_fused_op_trace()
    trace_probe_input = torch.randn(2, 16, device=device, dtype=dtype)
    trace_probe_weight = torch.randn(16, device=device, dtype=dtype)
    rmsnorm(trace_probe_input, trace_probe_weight)
    sglang_silu_and_mul(torch.randn(2, 16, device=device, dtype=dtype))
    trace = [
        {"op": record.op, "backend": record.backend}
        for record in get_fused_op_trace()
    ]
    clear_fused_op_trace()
    disable_fused_op_trace()

    for tokens, hidden, intermediate in CASES:
        torch.manual_seed(SEED + tokens)
        torch.cuda.reset_peak_memory_stats()
        input_tensor = torch.randn(tokens, hidden, device=device, dtype=dtype)
        norm_weight = torch.randn(hidden, device=device, dtype=dtype)
        gate_weight = (
            torch.randn(2 * intermediate, hidden, device=device, dtype=dtype)
            * hidden**-0.5
        )
        output_weight = (
            torch.randn(hidden, intermediate, device=device, dtype=dtype)
            * intermediate**-0.5
        )

        norm_result = rmsnorm(input_tensor, norm_weight)
        gate_result = linear_bf16_fp32(norm_result, gate_weight).to(dtype)
        activation_result = sglang_silu_and_mul(gate_result)
        chain_result = linear_bf16_fp32(
            activation_result, output_weight
        ).to(dtype)

        norm_reference = (
            input_tensor.float()
            * torch.rsqrt(
                input_tensor.float().pow(2).mean(dim=-1, keepdim=True) + EPS
            )
            * norm_weight.float()
        ).to(dtype)
        gate_reference = torch.matmul(norm_reference, gate_weight.t())
        activation_reference = (
            torch.nn.functional.silu(gate_reference.float()[..., :intermediate])
            * gate_reference.float()[..., intermediate:]
        ).to(dtype)
        chain_reference = torch.matmul(activation_reference, output_weight.t())

        norm_input_copy = input_tensor.clone()
        norm_output = torch.full_like(input_tensor, 123.0)
        norm_with_output = rmsnorm(
            norm_input_copy, norm_weight, EPS, out=norm_output
        )
        activation_input_copy = gate_result.clone()
        activation_output = torch.full(
            (tokens, intermediate), 7.0, device=device, dtype=dtype
        )
        activation_with_output = sglang_silu_and_mul(
            activation_input_copy, out=activation_output
        )
        gate_weight_copy = gate_weight.clone()
        projection_result = linear_bf16_fp32(norm_result, gate_weight)

        if args.backend == "aiter":
            projection_dispatch = {
                "projection_1": get_GEMM_A16W16_config(
                    tokens,
                    2 * intermediate,
                    hidden,
                    False,
                    str(dtype),
                    str(dtype),
                ),
                "projection_2": get_GEMM_A16W16_config(
                    tokens,
                    hidden,
                    intermediate,
                    False,
                    str(dtype),
                    str(dtype),
                ),
            }
        else:
            projection_dispatch = {
                "projection_1": {"libtype": "torch"},
                "projection_2": {"libtype": "torch"},
            }

        timings = {
            "rmsnorm": cuda_timing(lambda: rmsnorm(input_tensor, norm_weight)),
            "projection_1": cuda_timing(
                lambda: linear_bf16_fp32(norm_result, gate_weight)
            ),
            "silu_and_mul": cuda_timing(lambda: sglang_silu_and_mul(gate_result)),
            "projection_2": cuda_timing(
                lambda: linear_bf16_fp32(activation_result, output_weight)
            ),
            "complete_block": cuda_timing(
                lambda: linear_bf16_fp32(
                    sglang_silu_and_mul(
                        linear_bf16_fp32(
                            rmsnorm(input_tensor, norm_weight), gate_weight
                        ).to(dtype)
                    ),
                    output_weight,
                ).to(dtype)
            ),
        }
        operator_sum = sum(
            timings[name]["median_ms"]
            for name in ("rmsnorm", "projection_1", "silu_and_mul", "projection_2")
        )

        final_delta = difference(chain_result, chain_reference)
        results.append(
            {
                "case": {
                    "tokens": tokens,
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "dtype": str(dtype),
                },
                "weight_bytes": {
                    "norm": norm_weight.numel() * norm_weight.element_size(),
                    "gate": gate_weight.numel() * gate_weight.element_size(),
                    "output": output_weight.numel() * output_weight.element_size(),
                    "total": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in (norm_weight, gate_weight, output_weight)
                    ),
                },
                "dtype_contract": {
                    "rmsnorm": str(norm_result.dtype),
                    "projection_1_raw": str(
                        linear_bf16_fp32(norm_result, gate_weight).dtype
                    ),
                    "projection_1_chain_input": str(gate_result.dtype),
                    "silu_and_mul": str(activation_result.dtype),
                    "projection_2_raw": str(
                        linear_bf16_fp32(
                            activation_result, output_weight
                        ).dtype
                    ),
                    "complete_block": str(chain_result.dtype),
                },
                "mutation_contract": {
                    "rmsnorm_out_identity": norm_with_output is norm_output,
                    "rmsnorm_input_unchanged": torch.equal(
                        norm_input_copy, input_tensor
                    ),
                    "activation_out_identity": activation_with_output
                    is activation_output,
                    "activation_input_unchanged": torch.equal(
                        activation_input_copy, gate_result
                    ),
                    "projection_weight_unchanged": torch.equal(
                        gate_weight, gate_weight_copy
                    ),
                    "projection_returns_new_tensor": projection_result
                    is not norm_result,
                },
                "reference_differences": {
                    "rmsnorm": difference(norm_result, norm_reference),
                    "projection_1": difference(gate_result, gate_reference),
                    "silu_and_mul": difference(
                        activation_result, activation_reference
                    ),
                    "complete_block": final_delta,
                },
                "numerical_gate": {
                    "atol": 0.02,
                    "rtol": 0.02,
                    "method": "torch.allclose(atol=0.02, rtol=0.02)",
                    "passed": bool(
                        torch.allclose(
                            chain_result, chain_reference, atol=0.02, rtol=0.02
                        )
                    ),
                },
                "timings": timings,
                "projection_dispatch": projection_dispatch,
                "operator_sum_median_ms": operator_sum,
                "complete_minus_operator_sum_ms": timings["complete_block"][
                    "median_ms"
                ]
                - operator_sum,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            }
        )

    properties = torch.cuda.get_device_properties(0)
    output = {
        "label": "real dispatched backend benchmark on one MI300X",
        "campaign": "repo-e2e-20260909",
        "backend": args.backend,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd="/job/sglang", text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd="/job/sglang", text=True
            ).strip()
        ),
        "command": (
            f"/opt/venv/bin/python {Path(__file__)} --backend {args.backend} "
            f"--output {args.output}"
        ),
        "python": "/opt/venv/bin/python",
        "python_version": platform.python_version(),
        "torch": {
            "version": torch.__version__,
            "file": torch.__file__,
            "hip": str(torch.version.hip),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
        },
        "aiter_device": {"gfx": gfx, "cu_num": cu_num},
        "image_identity": {
            "local_image_id": "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
            "source": "task-specified qualified local image ID",
        },
        "backend_selection": {
            "fused_op_backend": get_fused_op_backend().value,
            "sglang_use_aiter": gemm_module._use_aiter,
            "trace": trace,
        },
        "native_paths": {
            "sglang": sglang.__file__,
            "rmsnorm_public_wrapper": source_path(rmsnorm),
            "activation_public_wrapper": source_path(sglang_silu_and_mul),
            "projection_public_wrapper": source_path(linear_bf16_fp32),
            "aiter_rmsnorm": source_path(rmsnorm2d_fwd),
            "aiter_activation": source_path(silu_and_mul),
            "aiter_tuned_gemm": source_path(tgemm.mm),
            "loaded_native_libraries": loaded_native_paths(),
            "torch_mm_cuda_dispatch": "aten::mm CUDA registered in RegisterCUDA_0.cpp",
        },
        "workloads": {"cases": CASES, "count": len(CASES)},
        "timing_method": {
            "clock": "CUDA events",
            "warmup_iterations": WARMUP_ITERATIONS,
            "timed_iterations": TIMED_ITERATIONS,
            "bounded": True,
        },
        "reference": (
            "independent Torch composition with the same bf16 inputs and weights; "
            "norm and activation use float32 intermediates before bf16 casts, "
            "while projections use bf16 Torch matmul"
        ),
        "raw_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
