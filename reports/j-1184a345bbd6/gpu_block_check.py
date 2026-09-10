#!/usr/bin/env python3
"""Bounded MI300X check for a norm/projection/activation block."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.layernorm import fused_add_rmsnorm
from sglang.srt.layers.linear import ReplicatedLinear


NUM_TOKENS = 16
HIDDEN_SIZE = 512
PROJECTION_SIZE = 2048
ACTIVATION_SIZE = PROJECTION_SIZE // 2
EPS = torch.finfo(torch.bfloat16).eps
WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 30
ATOL = 2e-2
RTOL = 2e-2


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _make_case(name: str, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(
        (NUM_TOKENS, HIDDEN_SIZE),
        generator=generator,
        dtype=torch.float32,
    )
    residual_base = torch.randn(
        (NUM_TOKENS, HIDDEN_SIZE),
        generator=generator,
        dtype=torch.float32,
    )

    if name == "zeros":
        values = torch.zeros_like(base)
        residual = torch.zeros_like(residual_base)
    elif name == "tiny_finite":
        values = base * 2**-10
        residual = residual_base * 2**-10
    elif name == "mixed_magnitudes":
        exponents = torch.randint(-4, 5, base.shape, generator=generator)
        values = base * torch.pow(2.0, exponents.float())
        residual = residual_base * torch.pow(
            2.0,
            torch.randint(-4, 5, residual_base.shape, generator=generator).float(),
        )
    elif name == "cancellation":
        values = base
        residual = -base + residual_base * 2**-8
    elif name == "sparse_skew":
        mask = torch.rand(base.shape, generator=generator) < 0.05
        values = base * 8.0 * mask
        residual_mask = torch.rand(
            residual_base.shape, generator=generator
        ) < 0.05
        residual = residual_base * 8.0 * residual_mask
    elif name == "random_baseline":
        values = base
        residual = residual_base
    else:
        raise ValueError(f"Unknown case: {name}")

    return {
        "input": values.to(torch.bfloat16).cuda(),
        "residual": residual.to(torch.bfloat16).cuda(),
    }


def _reference(
    case: dict[str, torch.Tensor],
    norm_weight: torch.Tensor,
    projection: ReplicatedLinear,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    accumulated = case["input"].float() + case["residual"].float()
    residual_out = accumulated.to(torch.bfloat16)
    variance = accumulated.pow(2).mean(dim=-1, keepdim=True)
    normed = (
        accumulated * torch.rsqrt(variance + EPS) * norm_weight.float()
    ).to(torch.bfloat16)
    projected = F.linear(normed, projection.weight)
    gate = projected[..., :ACTIVATION_SIZE].float()
    up = projected[..., ACTIVATION_SIZE:]
    activated = F.silu(gate).to(torch.bfloat16) * up
    return activated, residual_out, normed


def _sglang_block(
    case: dict[str, torch.Tensor],
    norm_weight: torch.Tensor,
    projection: ReplicatedLinear,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_tensor = case["input"].clone()
    residual = case["residual"].clone()
    fused_add_rmsnorm(input_tensor, residual, norm_weight, EPS)
    projected = projection(input_tensor)[0]
    activated = silu_and_mul(projected)
    return activated, residual, input_tensor


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs()
    relative = torch.where(
        denominator > 0,
        difference / denominator,
        difference,
    )
    return {
        "max_abs_error": difference.max().item(),
        "max_relative_error": relative.max().item(),
        "finite": bool(torch.isfinite(actual).all().item()),
        "matches_gate": bool(
            torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)
        ),
    }


def _time_cuda(
    setup: Callable[[], None] | None,
    operation: Callable[[], Any],
) -> dict[str, Any]:
    for _ in range(WARMUP_ITERATIONS):
        if setup is not None:
            setup()
        operation()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(MEASURED_ITERATIONS):
        if setup is not None:
            setup()
            torch.cuda.synchronize()
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))

    samples_ms.sort()
    return {
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations": MEASURED_ITERATIONS,
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p10_ms": samples_ms[math.floor(0.10 * len(samples_ms))],
        "p90_ms": samples_ms[math.floor(0.90 * len(samples_ms))],
    }


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def _backend_name(operator: Any) -> str:
    method = getattr(operator, "_forward_method", None)
    return getattr(method, "__name__", "unresolved")


def run(output_path: Path, repo: Path) -> None:
    torch.manual_seed(1184)
    norm_weight = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    projection = ReplicatedLinear(
        HIDDEN_SIZE,
        PROJECTION_SIZE,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        projection.weight.copy_(torch.randn_like(projection.weight) * 0.02)

    case_names = [
        "zeros",
        "tiny_finite",
        "mixed_magnitudes",
        "cancellation",
        "sparse_skew",
        "random_baseline",
    ]
    cases = {
        name: _make_case(name, 1184 + index)
        for index, name in enumerate(case_names)
    }

    correctness: dict[str, Any] = {}
    timing: dict[str, Any] = {}
    with torch.no_grad():
        for name, case in cases.items():
            actual, residual_actual, normed_actual = _sglang_block(
                case, norm_weight, projection
            )
            expected, residual_expected, normed_expected = _reference(
                case, norm_weight, projection
            )
            correctness[name] = {
                "complete_output": _error_metrics(actual, expected),
                "normed_output": _error_metrics(normed_actual, normed_expected),
                "residual_output": _error_metrics(residual_actual, residual_expected),
                "output_dtype": str(actual.dtype),
            }

            input_copy = case["input"].clone()
            residual_copy = case["residual"].clone()
            norm_timing = _time_cuda(
                lambda: (
                    input_copy.copy_(case["input"]),
                    residual_copy.copy_(case["residual"]),
                ),
                lambda: fused_add_rmsnorm(
                    input_copy, residual_copy, norm_weight, EPS
                ),
            )
            projection_timing = _time_cuda(
                None,
                lambda: projection(input_copy)[0],
            )
            activation_input = projection(input_copy)[0]
            activation_timing = _time_cuda(
                None,
                lambda: silu_and_mul(activation_input),
            )
            chain_timing = _time_cuda(
                lambda: (
                    input_copy.copy_(case["input"]),
                    residual_copy.copy_(case["residual"]),
                ),
                lambda: (
                    fused_add_rmsnorm(
                        input_copy, residual_copy, norm_weight, EPS
                    ),
                    silu_and_mul(projection(input_copy)[0]),
                )[1],
            )
            per_operator_sum_ms = (
                norm_timing["mean_ms"]
                + projection_timing["mean_ms"]
                + activation_timing["mean_ms"]
            )
            timing[name] = {
                "fused_add_rmsnorm": norm_timing,
                "projection": projection_timing,
                "silu_and_mul": activation_timing,
                "complete_block": chain_timing,
                "per_operator_sum_ms": per_operator_sum_ms,
                "chain_minus_operator_sum_ms": (
                    chain_timing["mean_ms"] - per_operator_sum_ms
                ),
            }

    with torch.no_grad():
        contract_case = cases["random_baseline"]
        input_tensor = contract_case["input"].clone()
        residual = contract_case["residual"].clone()
        input_before = input_tensor.clone()
        residual_before = residual.clone()
        norm_result = fused_add_rmsnorm(
            input_tensor, residual, norm_weight, EPS
        )
        norm_mutated = not torch.equal(
            input_tensor, input_before
        ) and not torch.equal(residual, residual_before)

        projection_input = input_tensor.clone()
        projection_input_before = projection_input.clone()
        projection_weight_before = projection.weight.clone()
        projected = projection(projection_input)[0]
        projection_unchanged = torch.equal(
            projection_input, projection_input_before
        ) and torch.equal(projection.weight, projection_weight_before)

        activation_input = projected.clone()
        activation_input_before = activation_input.clone()
        activation_out = torch.empty(
            (NUM_TOKENS, ACTIVATION_SIZE),
            dtype=torch.bfloat16,
            device="cuda",
        )
        activation_result = silu_and_mul(activation_input, activation_out)
        activation_contract = (
            activation_result is activation_out
            and torch.equal(activation_input, activation_input_before)
        )

    from sglang.kernels.ops.activation import _SILU_AND_MUL
    from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM

    result = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "commit": _git_output(repo, "rev-parse", "HEAD"),
            "branch": _git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "python_path": str(Path(torch.__file__).resolve()),
            "sglang_import": str(Path(__import__("sglang").__file__).resolve()),
        },
        "gpu": {
            "count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
        },
        "configuration": {
            "num_tokens": NUM_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "projection_size": PROJECTION_SIZE,
            "activation_size": ACTIVATION_SIZE,
            "dtype": "torch.bfloat16",
            "eps": EPS,
            "synthetic_tensor_bytes": _tensor_bytes(
                norm_weight,
                projection.weight,
                *(tensor for case in cases.values() for tensor in case.values()),
            ),
            "numerical_gate": {"atol": ATOL, "rtol": RTOL},
            "timing_method": (
                "CUDA events; 5 warmups and 30 measured launches; input resets "
                "are synchronized outside the timed interval"
            ),
        },
        "backends": {
            "fused_add_rmsnorm": _backend_name(_FUSED_ADD_RMSNORM),
            "silu_and_mul": _backend_name(_SILU_AND_MUL),
            "projection": (
                "ReplicatedLinear/UnquantizedLinearMethod with SGLANG_USE_AITER=1"
            ),
        },
        "contracts": {
            "fused_add_rmsnorm_returns_none": norm_result is None,
            "fused_add_rmsnorm_mutates_input_and_residual": norm_mutated,
            "projection_preserves_input_and_weight": projection_unchanged,
            "silu_and_mul_out_is_returned_and_input_unchanged": activation_contract,
            "all_output_dtypes_bfloat16": all(
                case_result["output_dtype"] == "torch.bfloat16"
                for case_result in correctness.values()
            ),
        },
        "correctness": correctness,
        "timing_ms": timing,
        "unsupported_boundaries": {
            "gemm.tiny_gemm_jit": (
                "test_tiny_gemm.py explicitly skips HIP runtime and requires SM90+"
            ),
            "sgl_kernel_dsv3_fused_a_gemm": (
                "not present in the installed sgl_kernel module on this HIP stack"
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/j-1184a345bbd6/results.json"),
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[2])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("One CUDA/HIP GPU is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one GPU must be visible")
    run(args.output, args.repo)


if __name__ == "__main__":
    main()
