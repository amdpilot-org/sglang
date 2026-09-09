#!/usr/bin/env python3
"""Measure ROCm RMSNorm static-FP8 producer-fusion dispatch and semantics."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch


IMAGE_ID = (
    "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5 "
    "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1"
)
FIXED_SCALE = 0.05
WARMUP_ITERATIONS = 25
TIMED_ITERATIONS = 200
TIMING_REPEATS = 5
SHAPES = ((128, 4096), (8192, 4096))

# These gates are fixed before measurement and are not adjusted during the run.
NUMERICAL_GATES = {
    "exact_fp8_bytes_equal_percent": 100.0,
    "dequant_max_abs_error": 0.10,
    "dequant_cosine": 0.999,
}


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def warm_time(
    function: Callable[[], Any],
    warmup_iterations: int,
    timed_iterations: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup_iterations):
        function()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed_iterations):
            function()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end) / timed_iterations)

    return {
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "repeats": repeats,
        "median_us": statistics_median(samples_ms) * 1000.0,
        "mean_us": sum(samples_ms) / len(samples_ms) * 1000.0,
        "min_us": min(samples_ms) * 1000.0,
        "max_us": max(samples_ms) * 1000.0,
        "samples_us": [sample * 1000.0 for sample in samples_ms],
    }


def statistics_median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def compare_outputs(
    fused_fp8: torch.Tensor,
    fused_scale: torch.Tensor,
    static_fp8: torch.Tensor,
    static_scale: torch.Tensor,
) -> dict[str, Any]:
    fused_bytes = fused_fp8.view(torch.uint8).flatten()
    static_bytes = static_fp8.view(torch.uint8).flatten()
    equal_mask = fused_bytes == static_bytes
    equal_count = int(equal_mask.sum().item())
    total = int(fused_bytes.numel())

    fused_dequant = fused_fp8.float() * fused_scale
    static_dequant = static_fp8.float() * static_scale
    difference = fused_dequant - static_dequant
    flat_difference = difference.flatten()
    flat_static = static_dequant.flatten()
    max_abs_error = float(flat_difference.abs().max().item())
    mean_abs_error = float(flat_difference.abs().mean().item())
    relative_l2 = float(
        (torch.linalg.vector_norm(flat_difference) / torch.linalg.vector_norm(flat_static))
        .item()
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            fused_dequant.flatten(), static_dequant.flatten(), dim=0
        ).item()
    )
    scale_equal = bool(torch.all(fused_scale == static_scale).item())

    return {
        "total_fp8_bytes": total,
        "exact_equal_fp8_bytes": equal_count,
        "exact_fp8_bytes_equal_percent": equal_count * 100.0 / total,
        "fused_scale_shape": list(fused_scale.shape),
        "static_scale_shape": list(static_scale.shape),
        "fused_scale_min": float(fused_scale.min().item()),
        "fused_scale_max": float(fused_scale.max().item()),
        "fused_scale_mean": float(fused_scale.mean().item()),
        "static_scale": float(static_scale.reshape(-1)[0].item()),
        "returned_scale_exactly_static": scale_equal,
        "dequant_max_abs_error": max_abs_error,
        "dequant_mean_abs_error": mean_abs_error,
        "dequant_relative_l2": relative_l2,
        "dequant_cosine": cosine,
        "gates": {
            "exact_fp8_bytes_equal_pass": equal_count == total,
            "dequant_max_abs_error_pass": max_abs_error
            <= NUMERICAL_GATES["dequant_max_abs_error"],
            "dequant_cosine_pass": cosine >= NUMERICAL_GATES["dequant_cosine"],
            "returned_scale_exactly_static_pass": scale_equal,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/j-45813c577c41/results.json"),
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/HIP device is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one assigned GPU, got {torch.cuda.device_count()}")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    device_name = torch.cuda.get_device_name(device)
    if capability != (9, 4) or "MI300X" not in device_name:
        raise RuntimeError(f"Expected one gfx942 MI300X, got {capability} {device_name}")

    import aiter
    import sglang
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.layers.communicator import _fused_rmsnorm_fp8_per_token_quant
    from sglang.srt.layers.layernorm import (
        RMSNorm,
        _flashinfer_rmsnorm_quant_available,
    )
    import sglang.srt.layers.layernorm as layernorm_module

    torch.manual_seed(0)
    results: dict[str, Any] = {
        "campaign": "repo-e2e-20260909",
        "task": "j-45813c577c41",
        "producer": "RMSNorm",
        "source_commit": git_output(repo, "rev-parse", "HEAD"),
        "source_branch": git_output(repo, "branch", "--show-current"),
        "source_dirty": bool(git_output(repo, "status", "--porcelain")),
        "image_identity": IMAGE_ID,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "gpu": {
            "name": device_name,
            "capability": list(capability),
            "device_count": torch.cuda.device_count(),
        },
        "source_paths": {
            "sglang": sglang.__file__,
            "layernorm": layernorm_module.__file__,
            "aiter_python": aiter.__file__,
            "aiter_native_rmsnorm_quant": "/sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so",
        },
        "fixed_scale": FIXED_SCALE,
        "warm_timing": {
            "warmup_iterations": WARMUP_ITERATIONS,
            "timed_iterations": TIMED_ITERATIONS,
            "repeats": TIMING_REPEATS,
            "method": "torch.cuda.Event around a loop; per-call value includes wrapper allocations and launch overhead",
        },
        "numerical_gates": NUMERICAL_GATES,
        "static_fusion_dispatch": {
            "flashinfer_rmsnorm_quant_available": _flashinfer_rmsnorm_quant_available,
            "currently_supported": _flashinfer_rmsnorm_quant_available,
        },
        "shapes": [],
    }

    for num_tokens, hidden_size in SHAPES:
        torch.manual_seed(0)
        layer = RMSNorm(hidden_size).to(device=device, dtype=torch.bfloat16)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        epsilon = float(layer.variance_epsilon)
        weight = layer.weight.data
        hidden_states = torch.randn(
            num_tokens, hidden_size, device=device, dtype=torch.bfloat16
        )
        residual = torch.randn_like(hidden_states)
        static_scale = torch.tensor(
            [FIXED_SCALE], device=device, dtype=torch.float32
        )

        with torch.inference_mode():
            separate_normed, separate_residual = layer(
                hidden_states.clone(), residual.clone()
            )
            static_fp8, returned_static_scale = static_quant_fp8(
                separate_normed, static_scale
            )

            fused_result, fused_residual = _fused_rmsnorm_fp8_per_token_quant(
                hidden_states.clone(),
                weight,
                epsilon,
                residual=residual.clone(),
            )
            fused_fp8, fused_scale = fused_result

            comparison = compare_outputs(
                fused_fp8,
                fused_scale,
                static_fp8,
                returned_static_scale,
            )

            original_scale_lookup = layernorm_module._fp8_static_input_scale
            layernorm_module._fp8_static_input_scale = lambda linear: static_scale
            try:
                production_output = layer(
                    hidden_states.clone(),
                    residual.clone(),
                    quant_linear=object(),
                )
            finally:
                layernorm_module._fp8_static_input_scale = original_scale_lookup

            try:
                layer.forward_with_per_tensor_quant_fusion(
                    hidden_states.clone(), static_scale, residual.clone()
                )
                direct_static_result = {"dispatched": True}
            except Exception as error:
                direct_static_result = {
                    "dispatched": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }

            timing_residual = residual.clone()
            fused_timing = warm_time(
                lambda: _fused_rmsnorm_fp8_per_token_quant(
                    hidden_states,
                    weight,
                    epsilon,
                    residual=timing_residual,
                ),
                WARMUP_ITERATIONS,
                TIMED_ITERATIONS,
                TIMING_REPEATS,
            )

            timing_residual = residual.clone()
            separate_timing_residual = residual.clone()

            def separate_path() -> None:
                normed, _ = layer(hidden_states, separate_timing_residual)
                static_quant_fp8(normed, static_scale)

            separate_timing = warm_time(
                separate_path,
                WARMUP_ITERATIONS,
                TIMED_ITERATIONS,
                TIMING_REPEATS,
            )

        shape_result = {
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "dtype": str(hidden_states.dtype),
            "fp8_dtype": str(static_fp8.dtype),
            "comparison": comparison,
            "production_dispatch": {
                "output_type": str(production_output[0].dtype),
                "is_prequantized_tuple": isinstance(production_output[0], tuple),
                "fell_back_to_unquantized_bf16": not isinstance(
                    production_output[0], tuple
                ),
            },
            "direct_static_fusion_call": direct_static_result,
            "timing_us": {
                "fused_rocm_per_token_rmsnorm_fp8": fused_timing,
                "separate_rmsnorm_then_static_fp8": separate_timing,
                "separate_minus_fused_median_us": separate_timing["median_us"]
                - fused_timing["median_us"],
            },
        }
        results["shapes"].append(shape_result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
