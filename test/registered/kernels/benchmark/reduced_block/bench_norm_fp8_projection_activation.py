from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from aiter import gemm_a8w8
from aiter.utility import dtypes as aiter_dtypes
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import fp8_scaled_mm
from sglang.kernels.ops.layernorm import rmsnorm


CASES = [2048, 4096, 8192, 16384, 32768, 65536]
HIDDEN = 4096
PROJECTION_OUT = 8192
WARMUP = 3
ITERATIONS = 20
ACCURACY_GATE = 0.02
WEIGHT_LIMIT_BYTES = 4 * 2**30
LIVE_LIMIT_BYTES = 48 * 2**30


def _git_root() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    except Exception:
        return str(Path(__file__).resolve().parents[5])


def _time_cuda(fn, *, warmup: int = WARMUP, iterations: int = ITERATIONS) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return {
        "warmup": warmup,
        "iterations": iterations,
        "total_ms": total_ms,
        "mean_ms": total_ms / iterations,
    }


def _quant_rows(value: torch.Tensor, fp8_dtype: torch.dtype):
    scale = (
        value.float()
        .abs()
        .amax(dim=1, keepdim=True)
        .clamp_min(1e-12)
        / torch.finfo(fp8_dtype).max
    )
    return (value.float() / scale).to(fp8_dtype), scale


def _reference_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = value.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + 1e-6) * weight.float()


def _reference_projection(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(
        quantized.float() * scale,
        (weight.float() * weight_scale.t()).t(),
    )


def _reference_activation(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.nn.functional.silu(value[..., :half]) * value[..., half:]


def _module_path(name: str) -> str:
    module = __import__(name)
    return str(getattr(module, "__file__", name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires exactly one CUDA/HIP GPU")

    torch.manual_seed(29630)
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    fp8_dtype = aiter_dtypes.fp8
    root = _git_root()
    started = time.perf_counter()

    public_projection_error = None
    try:
        probe_scale_a = torch.ones(1, 1, device=device, dtype=torch.float32)
        probe_scale_b = torch.ones(16, device=device, dtype=torch.float32)
        probe_a = torch.zeros(1, 16, device=device, dtype=fp8_dtype)
        probe_b = torch.zeros(16, 16, device=device, dtype=fp8_dtype).t().contiguous()
        fp8_scaled_mm(
            probe_a,
            probe_b,
            probe_scale_a,
            probe_scale_b,
            torch.bfloat16,
        )
    except Exception as exc:
        public_projection_error = f"{type(exc).__name__}: {exc}"
    finally:
        del probe_scale_a, probe_scale_b, probe_a, probe_b

    weight_bf16 = torch.randn(
        PROJECTION_OUT, HIDDEN, device=device, dtype=torch.bfloat16
    )
    norm_weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16)
    weight_scale = (
        weight_bf16.float()
        .abs()
        .amax(dim=1, keepdim=True)
        .clamp_min(1e-12)
        / torch.finfo(fp8_dtype).max
    )
    projection_weight = (weight_bf16.float() / weight_scale).to(fp8_dtype).contiguous()
    weight_scale = weight_scale.t().contiguous()
    weight_bytes = projection_weight.numel() * projection_weight.element_size()
    if weight_bytes >= WEIGHT_LIMIT_BYTES:
        raise RuntimeError(f"Weight allocation exceeds 4 GiB: {weight_bytes}")

    cases = []
    for case_index, tokens in enumerate(CASES):
        torch.cuda.reset_peak_memory_stats()
        case_started = time.perf_counter()
        x = torch.randn(tokens, HIDDEN, device=device, dtype=torch.bfloat16)
        x_before = x.clone()

        def chain():
            normed = rmsnorm(x, norm_weight, eps=1e-6)
            quantized, scale = _quant_rows(normed, fp8_dtype)
            projected = gemm_a8w8(
                quantized,
                projection_weight,
                scale,
                weight_scale,
                dtype=torch.bfloat16,
            )
            return silu_and_mul(projected)

        cold_started = time.perf_counter()
        actual = chain()
        torch.cuda.synchronize()
        cold_seconds = time.perf_counter() - cold_started

        normed = rmsnorm(x, norm_weight, eps=1e-6)
        quantized, scale = _quant_rows(normed, fp8_dtype)
        projected = gemm_a8w8(
            quantized,
            projection_weight,
            scale,
            weight_scale,
            dtype=torch.bfloat16,
        )

        reference_normed = _reference_norm(x, norm_weight)
        reference_quantized, reference_scale = _quant_rows(
            reference_normed, fp8_dtype
        )
        reference_projected = _reference_projection(
            reference_quantized,
            reference_scale,
            projection_weight,
            weight_scale,
        ).to(torch.bfloat16)
        expected = _reference_activation(reference_projected.float()).to(
            torch.bfloat16
        )

        absolute_error = (actual.float() - expected.float()).abs()
        normalized_rmse = (
            absolute_error.pow(2).mean().sqrt()
            / expected.float().pow(2).mean().sqrt()
        ).item()

        norm_out = torch.empty_like(normed)
        rmsnorm(x, norm_weight, eps=1e-6, out=norm_out)
        activation_out = torch.empty_like(actual)
        silu_and_mul(projected, out=activation_out)

        norm_timing = _time_cuda(lambda: rmsnorm(x, norm_weight, eps=1e-6))
        def projection():
            quantized_input, input_scale = _quant_rows(
                rmsnorm(x, norm_weight, eps=1e-6), fp8_dtype
            )
            return gemm_a8w8(
                quantized_input,
                projection_weight,
                input_scale,
                weight_scale,
                dtype=torch.bfloat16,
            )

        projection_timing = _time_cuda(projection)
        activation_timing = _time_cuda(lambda: silu_and_mul(projected))
        chain_timing = _time_cuda(chain)

        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        if peak_allocated >= LIVE_LIMIT_BYTES:
            raise RuntimeError(f"Live allocation exceeds 48 GiB: {peak_allocated}")

        cases.append(
            {
                "case_index": case_index,
                "tokens": tokens,
                "hidden": HIDDEN,
                "projection_out": PROJECTION_OUT,
                "cold_first_chain_seconds": cold_seconds,
                "warm": {
                    "norm": norm_timing,
                    "projection_quant_gemm": projection_timing,
                    "activation": activation_timing,
                    "whole_chain": chain_timing,
                },
                "useful_throughput_tokens_per_second": tokens
                / (chain_timing["mean_ms"] / 1000),
                "projection_tflops": (
                    2
                    * tokens
                    * HIDDEN
                    * PROJECTION_OUT
                    / (projection_timing["mean_ms"] / 1000)
                    / 1e12
                ),
                "accuracy": {
                    "normalized_rmse": normalized_rmse,
                    "gate": "normalized_rmse <= 0.02",
                    "passed": normalized_rmse <= ACCURACY_GATE,
                },
                "contracts": {
                    "actual_dtype": str(actual.dtype),
                    "expected_dtype": str(expected.dtype),
                    "dtype_passed": actual.dtype == expected.dtype,
                    "input_unchanged": bool(torch.equal(x, x_before)),
                    "norm_explicit_out_passed": bool(
                        torch.equal(norm_out, normed)
                    ),
                    "activation_explicit_out_passed": bool(
                        torch.equal(activation_out, actual)
                    ),
                },
                "memory": {
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                },
                "case_wall_seconds": time.perf_counter() - case_started,
            }
        )

        del (
            x,
            x_before,
            actual,
            normed,
            quantized,
            scale,
            projected,
            reference_normed,
            reference_quantized,
            reference_scale,
            reference_projected,
            expected,
            absolute_error,
            norm_out,
            activation_out,
        )
        torch.cuda.empty_cache()

    result = {
        "label": "MI300X reduced-block norm -> FP8 projection -> gated activation study",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_root": root,
        "python": sys.executable,
        "torch_version": torch.__version__,
        "torch_path": _module_path("torch"),
        "sglang_path": _module_path("sglang"),
        "sgl_kernel_path": _module_path("sgl_kernel"),
        "aiter_path": _module_path("aiter"),
        "gpu": {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        },
        "operators": {
            "norm": "sglang.kernels.ops.layernorm.rmsnorm (Aiter backend on HIP)",
            "activation": "sglang.kernels.ops.activation.silu_and_mul (AOT backend)",
            "projection_control": "aiter.gemm_a8w8 with per-token/per-row FP8 scales",
            "public_projection": "sglang.kernels.ops.gemm.fp8_scaled_mm",
            "public_projection_error": public_projection_error,
        },
        "dimensions": {
            "token_cases": CASES,
            "hidden": HIDDEN,
            "projection_out": PROJECTION_OUT,
        },
        "dtypes": {
            "input": "torch.bfloat16",
            "norm_weight": "torch.bfloat16",
            "projection_fp8": str(fp8_dtype),
            "scales": "torch.float32",
            "output": "torch.bfloat16",
        },
        "timing_method": {
            "cold": "one synchronized chain call per case",
            "warm": f"{WARMUP} warmups and {ITERATIONS} timed calls per operator and chain",
            "clock": "CUDA/HIP events",
        },
        "limits": {
            "max_cases": len(CASES),
            "weight_limit_bytes": WEIGHT_LIMIT_BYTES,
            "live_allocation_limit_bytes": LIVE_LIMIT_BYTES,
            "accuracy_gate_normalized_rmse": ACCURACY_GATE,
        },
        "weight_bytes": weight_bytes,
        "cases": cases,
        "all_accuracy_gates_passed": all(
            case["accuracy"]["passed"] for case in cases
        ),
        "all_contracts_passed": all(
            all(case["contracts"].values()) for case in cases
        ),
        "total_wall_seconds": time.perf_counter() - started,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
