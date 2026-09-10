"""Bounded chunk-partition benchmark for a reduced SGLang MLP block.

The block uses the public ``fused_add_rmsnorm`` and ``silu_and_mul`` operators.
The projection boundary is explicit: ``sglang.kernels.ops.gemm`` has no
HIP-eligible backend on gfx942, so Torch ``F.linear`` is the projection control.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import _FP8_SCALED_MM
from sglang.kernels.ops.layernorm import fused_add_rmsnorm
from sglang.kernels.spec import PlatformInfo, capabilities_satisfied


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 8192
EPSILON = 1e-6
DTYPE = torch.bfloat16
WARMUP_ITERATIONS = 3
MEASUREMENT_ITERATIONS = 10
ATOL = 2e-2
RTOL = 2e-2

WORKLOAD_CASES = (
    (1024, None),
    (1024, (256, 512, 256)),
    (4096, None),
    (4096, (1024, 2048, 1024)),
    (16384, None),
    (16384, (4096, 8192, 4096)),
)


def _git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def _split_rows(tensor: torch.Tensor, partitions: Sequence[int] | None):
    if partitions is None:
        return (tensor,)
    return torch.split(tensor, list(partitions), dim=0)


def _run_block(
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    partitions: Sequence[int] | None,
) -> torch.Tensor:
    input_chunks = _split_rows(input_tensor, partitions)
    residual_chunks = _split_rows(residual_tensor, partitions)
    output_chunks = []
    for input_chunk, residual_chunk in zip(input_chunks, residual_chunks):
        fused_add_rmsnorm(input_chunk, residual_chunk, norm_weight, EPSILON)
        projected = F.linear(input_chunk, up_weight)
        activated = silu_and_mul(projected)
        output_chunks.append(F.linear(activated, down_weight))
    return output_chunks[0] if len(output_chunks) == 1 else torch.cat(output_chunks)


def _torch_reference(
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    accumulated = input_tensor.to(torch.float32) + residual_tensor.to(torch.float32)
    residual_reference = accumulated.to(DTYPE)
    variance = accumulated.pow(2).mean(dim=-1, keepdim=True)
    normed = accumulated * torch.rsqrt(variance + EPSILON)
    normed_reference = (normed * norm_weight.to(torch.float32)).to(DTYPE)
    projected = F.linear(normed_reference, up_weight)
    gate = projected[..., :INTERMEDIATE_SIZE]
    up = projected[..., INTERMEDIATE_SIZE:]
    activated = F.silu(gate) * up
    output_reference = F.linear(activated, down_weight)
    return output_reference, normed_reference, residual_reference


def _time_body(
    setup: Callable[[], object],
    body: Callable[[object], object],
) -> dict[str, float]:
    for _ in range(WARMUP_ITERATIONS):
        arguments = setup()
        body(arguments)
        del arguments
    torch.cuda.synchronize()

    elapsed_seconds = []
    for _ in range(MEASUREMENT_ITERATIONS):
        arguments = setup()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        body(arguments)
        end_event.record()
        end_event.synchronize()
        elapsed_seconds.append(start_event.elapsed_time(end_event) / 1000.0)
        del arguments

    return {
        "mean_seconds": statistics.fmean(elapsed_seconds),
        "median_seconds": statistics.median(elapsed_seconds),
        "min_seconds": min(elapsed_seconds),
        "max_seconds": max(elapsed_seconds),
        "iterations": len(elapsed_seconds),
        "raw_seconds": elapsed_seconds,
    }


def _peak_memory(
    setup: Callable[[], object],
    body: Callable[[object], object],
) -> dict[str, float]:
    arguments = setup()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    body(arguments)
    torch.cuda.synchronize()
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _operator_timings(
    base_input: torch.Tensor,
    base_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    partitions: Sequence[int] | None,
) -> dict[str, object]:
    normed_input = base_input.clone()
    residual_input = base_residual.clone()
    fused_add_rmsnorm(normed_input, residual_input, norm_weight, EPSILON)
    projected_input = F.linear(normed_input, up_weight)
    activated_input = silu_and_mul(projected_input)

    def setup_norm():
        if partitions is None:
            return (base_input.clone(), base_residual.clone())
        return (
            list(_split_rows(base_input.clone(), partitions)),
            list(_split_rows(base_residual.clone(), partitions)),
        )

    def body_norm(arguments):
        input_chunks, residual_chunks = arguments
        if partitions is None:
            fused_add_rmsnorm(input_chunks, residual_chunks, norm_weight, EPSILON)
        else:
            for input_chunk, residual_chunk in zip(input_chunks, residual_chunks):
                fused_add_rmsnorm(input_chunk, residual_chunk, norm_weight, EPSILON)

    def setup_projection():
        return _split_rows(normed_input, partitions)

    def body_up_projection(arguments):
        for input_chunk in arguments:
            F.linear(input_chunk, up_weight)

    def setup_activation():
        return _split_rows(projected_input, partitions)

    def body_activation(arguments):
        for projected_chunk in arguments:
            silu_and_mul(projected_chunk)

    def setup_down_projection():
        return _split_rows(activated_input, partitions)

    def body_down_projection(arguments):
        for activated_chunk in arguments:
            F.linear(activated_chunk, down_weight)

    return {
        "fused_add_rmsnorm": _time_body(setup_norm, body_norm),
        "up_projection": _time_body(setup_projection, body_up_projection),
        "silu_and_mul": _time_body(setup_activation, body_activation),
        "down_projection": _time_body(setup_down_projection, body_down_projection),
    }


def _projection_boundary() -> dict[str, object]:
    platform = PlatformInfo.detect()
    backends = []
    for backend in _FP8_SCALED_MM.priority:
        capabilities = _FP8_SCALED_MM.capabilities.get(backend, frozenset())
        backends.append(
            {
                "backend": backend.value,
                "capabilities": [str(capability) for capability in capabilities],
                "eligible_on_platform": capabilities_satisfied(capabilities, platform),
            }
        )
    return {
        "public_operator": "sglang.kernels.ops.gemm.fp8_scaled_mm",
        "platform_device": platform.device_type,
        "backends": backends,
        "boundary": (
            "No sglang.kernels.ops.gemm projection backend is HIP-eligible on "
            "gfx942; Torch F.linear is used only as the explicit projection control."
        ),
    }


def _case_result(
    token_count: int,
    partitions: Sequence[int] | None,
    base_input: torch.Tensor,
    base_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> dict[str, object]:
    input_tensor = base_input.clone()
    residual_tensor = base_residual.clone()
    cold_started = time.perf_counter()
    output = _run_block(
        input_tensor,
        residual_tensor,
        norm_weight,
        up_weight,
        down_weight,
        partitions,
    )
    torch.cuda.synchronize()
    cold_wall_seconds = time.perf_counter() - cold_started

    reference_output, reference_normed, reference_residual = _torch_reference(
        base_input,
        base_residual,
        norm_weight,
        up_weight,
        down_weight,
    )
    torch.testing.assert_close(
        input_tensor,
        reference_normed,
        atol=ATOL,
        rtol=RTOL,
        msg="final normed state versus independent Torch composition",
    )
    torch.testing.assert_close(
        residual_tensor,
        reference_residual,
        atol=ATOL,
        rtol=RTOL,
        msg="final residual state versus independent Torch composition",
    )

    projected_for_activation = F.linear(input_tensor, up_weight)
    projected_before_activation = projected_for_activation.clone()
    activated_sglang = silu_and_mul(projected_for_activation)
    activated_reference = F.silu(
        projected_for_activation[..., :INTERMEDIATE_SIZE]
    ) * projected_for_activation[..., INTERMEDIATE_SIZE:]
    torch.testing.assert_close(
        activated_sglang,
        activated_reference,
        atol=ATOL,
        rtol=RTOL,
        msg="silu_and_mul versus independent Torch composition on identical input",
    )
    torch.testing.assert_close(
        projected_for_activation,
        projected_before_activation,
        msg="silu_and_mul must not mutate its input",
    )

    if partitions is None:
        baseline_output = output
        baseline_normed = input_tensor
        baseline_residual = residual_tensor
    else:
        baseline_input = base_input.clone()
        baseline_residual_tensor = base_residual.clone()
        baseline_output = _run_block(
            baseline_input,
            baseline_residual_tensor,
            norm_weight,
            up_weight,
            down_weight,
            None,
        )
        baseline_normed = baseline_input
        baseline_residual = baseline_residual_tensor
    torch.testing.assert_close(
        output,
        baseline_output,
        atol=ATOL,
        rtol=RTOL,
        msg="chunked output versus unchunked output",
    )
    torch.testing.assert_close(
        input_tensor,
        baseline_normed,
        atol=ATOL,
        rtol=RTOL,
        msg="chunked final normed state versus unchunked final state",
    )
    torch.testing.assert_close(
        residual_tensor,
        baseline_residual,
        atol=ATOL,
        rtol=RTOL,
        msg="chunked final residual state versus unchunked final state",
    )

    output_difference = (output.float() - reference_output.float()).abs()
    output_reference_abs = reference_output.float().abs().clamp_min(1e-6)
    output_relative_error = output_difference / output_reference_abs

    def setup_chain():
        return (base_input.clone(), base_residual.clone())

    def body_chain(arguments):
        input_tensor, residual_tensor = arguments
        return _run_block(
            input_tensor,
            residual_tensor,
            norm_weight,
            up_weight,
            down_weight,
            partitions,
        )

    chain_timing = _time_body(setup_chain, body_chain)
    chain_memory = _peak_memory(setup_chain, body_chain)
    operator_timings = _operator_timings(
        base_input,
        base_residual,
        norm_weight,
        up_weight,
        down_weight,
        partitions,
    )

    operator_sum = sum(
        timing["mean_seconds"] for timing in operator_timings.values()
    )
    flops = 6 * token_count * HIDDEN_SIZE * INTERMEDIATE_SIZE
    mean_seconds = chain_timing["mean_seconds"]
    projected_bytes = (
        4 * token_count * HIDDEN_SIZE
        + 2 * token_count * INTERMEDIATE_SIZE
        + 2 * token_count * HIDDEN_SIZE
        + 2 * HIDDEN_SIZE * INTERMEDIATE_SIZE
        + 2 * INTERMEDIATE_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + INTERMEDIATE_SIZE
    ) * torch.empty((), dtype=DTYPE).element_size()

    return {
        "case_id": f"tokens-{token_count}-{'unchunked' if partitions is None else 'chunked'}",
        "token_count": token_count,
        "partitions_rows": list(partitions) if partitions is not None else None,
        "dtype": str(DTYPE),
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "cold_wall_seconds": cold_wall_seconds,
        "warm_chain": chain_timing,
        "operator_timings": operator_timings,
        "operator_shares": {
            operator_name: timing["mean_seconds"] / operator_sum
            for operator_name, timing in operator_timings.items()
        },
        "operator_sum_mean_seconds": operator_sum,
        "chain_minus_operator_sum_seconds": mean_seconds - operator_sum,
        "useful_tokens": token_count,
        "useful_flops": flops,
        "approximate_io_bytes": projected_bytes,
        "tokens_per_second": token_count / mean_seconds,
        "tflops": flops / mean_seconds / 1e12,
        "memory": chain_memory,
        "numerical_comparison": {
            "per_operator_method": (
                "torch.testing.assert_close against independent Torch composition "
                "on identical operator input"
            ),
            "chunked_vs_unchunked_method": (
                "torch.testing.assert_close for output and final state"
            ),
            "atol": ATOL,
            "rtol": RTOL,
            "whole_chain_independent_observation": {
                "max_abs_error": output_difference.max().item(),
                "mean_abs_error": output_difference.mean().item(),
                "max_relative_error": output_relative_error.max().item(),
                "mean_relative_error": output_relative_error.mean().item(),
                "note": (
                    "Accumulated bf16 projection/activation differences are "
                    "reported without changing the per-operator gate."
                ),
            },
            "output_dtype": str(output.dtype),
            "final_normed_dtype": str(input_tensor.dtype),
            "final_residual_dtype": str(residual_tensor.dtype),
            "mutation_contract": {
                "fused_add_rmsnorm": "mutates input and residual in place",
                "silu_and_mul": "returns a new tensor and leaves input unchanged",
            },
        },
    }


def run_benchmark(output_path: Path) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires exactly one CUDA/HIP GPU")

    torch.manual_seed(20260910)
    device_properties = torch.cuda.get_device_properties(0)
    results = {
        "label": "mirror-checkout chunk-partition benchmark",
        "git_commit": _git_commit(Path(__file__).resolve().parents[5]),
        "command": [sys.executable, *sys.argv],
        "gpu": {
            "name": device_properties.name,
            "total_memory_bytes": device_properties.total_memory,
            "device_count": torch.cuda.device_count(),
        },
        "torch_version": torch.__version__,
        "torch_hip": torch.version.hip,
        "source_paths": {
            "torch": torch.__file__,
            "sglang": __import__("sglang").__file__,
            "sgl_kernel": __import__("sgl_kernel").__file__,
            "aiter": __import__("aiter").__file__,
        },
        "native_paths": {
            "sgl_kernel": __import__("sgl_kernel").common_ops.__file__,
            "aiter_rmsnorm": __import__("aiter").jit.module_aiter_core.__file__,
        },
        "image_identity": {
            "qualified_image": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
            "local_image_id": "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
        },
        "limits": {
            "workload_cases": len(WORKLOAD_CASES),
            "warmup_iterations_per_measurement": WARMUP_ITERATIONS,
            "measurement_iterations_per_measurement": MEASUREMENT_ITERATIONS,
            "synthetic_weights_bytes": (
                2 * INTERMEDIATE_SIZE * HIDDEN_SIZE
                + HIDDEN_SIZE * INTERMEDIATE_SIZE
            )
            * torch.empty((), dtype=DTYPE).element_size()
        },
        "projection_boundary": _projection_boundary(),
        "timing_method": "CUDA/HIP events after three warmups; ten finite measurements",
        "cases": [],
    }

    for token_count, partitions in WORKLOAD_CASES:
        base_input = torch.randn(
            token_count, HIDDEN_SIZE, dtype=DTYPE, device="cuda"
        )
        base_residual = torch.randn_like(base_input)
        norm_weight = torch.randn(HIDDEN_SIZE, dtype=DTYPE, device="cuda")
        up_weight = torch.randn(
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            dtype=DTYPE,
            device="cuda",
        ) * 0.02
        down_weight = torch.randn(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            dtype=DTYPE,
            device="cuda",
        ) * 0.02
        case_result = _case_result(
            token_count,
            partitions,
            base_input,
            base_residual,
            norm_weight,
            up_weight,
            down_weight,
        )
        results["cases"].append(case_result)
        print(
            f"{case_result['case_id']}: "
            f"{case_result['tokens_per_second']:.2f} tokens/s, "
            f"{case_result['tflops']:.2f} TFLOP/s",
            flush=True,
        )
        del base_input, base_residual, norm_weight, up_weight, down_weight
        torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chunk-partition-results.json"),
        help="path for the raw JSON result",
    )
    arguments = parser.parse_args()
    run_benchmark(arguments.output)


if __name__ == "__main__":
    main()
