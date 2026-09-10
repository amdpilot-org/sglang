#!/usr/bin/env python3
"""Bounded MI300X reduced-block benchmark for public SGLang operators."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.gemm import dsv3_fused_a_gemm, fp8_scaled_mm
from sglang.kernels.ops.kvcache import reshape_and_cache_flash
from sglang.kernels.ops.layernorm import fused_add_rmsnorm


DTYPE = torch.bfloat16
HIDDEN_SIZE = 2048
ATTENTION_HEADS = 8
HEAD_SIZE = 128
ATTENTION_DIM = ATTENTION_HEADS * HEAD_SIZE
INTERMEDIATE_SIZE = 4096
BLOCK_SIZE = 16
WARMUP_STEPS = 3
TIMED_STEPS = 12
CHAIN_WARMUP_STEPS = 3
CHAIN_TIMED_STEPS = 12
CASES = (
    (1, 64),
    (1, 256),
    (4, 64),
    (4, 256),
    (16, 64),
    (16, 256),
)
MEMORY_LIMIT_BYTES = 48 * 1024**3
WEIGHT_LIMIT_BYTES = 4 * 1024**3
NUMERICAL_GATES = {
    "norm": {"rtol": 2e-2, "atol": 2e-2},
    "projection": {"rtol": 1e-2, "atol": 1e-2},
    "activation": {"rtol": 2e-2, "atol": 2e-2},
    "cache": {"rtol": 0.0, "atol": 0.0},
    "whole_chain": {"normalized_rmse_max": 0.02},
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * (len(ordered) - 1))))
    return ordered[index]


def allocator_block_bytes(logical_bytes: int) -> int:
    return ((logical_bytes + 511) // 512) * 512


def timing_stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def measure_cuda_operation(operation, warmup_steps: int, timed_steps: int) -> dict[str, float]:
    for _ in range(warmup_steps):
        operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(timed_steps):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end))
    return timing_stats(samples)


def comparison(actual: torch.Tensor, expected: torch.Tensor, gate: dict[str, float]) -> dict[str, object]:
    difference = actual.float() - expected.float()
    return {
        "allclose": bool(
            torch.allclose(
                actual,
                expected,
                rtol=gate["rtol"],
                atol=gate["atol"],
            )
        ),
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
    }


def normalized_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    return float(
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected.float())
    )


def reference_block(
    input_before: torch.Tensor,
    residual_before: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    residual_expected = (input_before.float() + residual_before.float()).to(DTYPE)
    normed_float = residual_expected.float()
    variance = normed_float.pow(2).mean(dim=-1, keepdim=True)
    normed_expected = (
        normed_float * torch.rsqrt(variance + 1e-6) * norm_weight.float()
    ).to(DTYPE)
    qkv_expected = F.linear(normed_expected, qkv_weight)
    gate_expected = F.linear(normed_expected, gate_weight)
    activated_expected = (
        F.silu(gate_expected[:, :INTERMEDIATE_SIZE].float())
        * gate_expected[:, INTERMEDIATE_SIZE:].float()
    ).to(DTYPE)
    down_expected = F.linear(activated_expected, down_weight)
    return residual_expected, normed_expected, qkv_expected, gate_expected, activated_expected, down_expected



def block_forward(
    input_state: torch.Tensor,
    residual_state: torch.Tensor,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    fused_add_rmsnorm(input_state, residual_state, norm_weight, 1e-6)
    qkv_output = F.linear(input_state, qkv_weight)
    query_offset = ATTENTION_DIM
    key_offset = 2 * ATTENTION_DIM
    key_source = qkv_output[:, query_offset:key_offset].reshape(
        input_state.shape[0], ATTENTION_HEADS, HEAD_SIZE
    )
    value_source = qkv_output[:, key_offset:].reshape(
        input_state.shape[0], ATTENTION_HEADS, HEAD_SIZE
    )
    reshape_and_cache_flash(
        key_source,
        value_source,
        key_cache,
        value_cache,
        slot_mapping,
    )
    gate_output = F.linear(input_state, gate_weight)
    activated_output = silu_and_mul(gate_output)
    down_output = F.linear(activated_output, down_weight)
    return qkv_output, gate_output, activated_output, down_output


def operator_paths() -> dict[str, str | None]:
    paths = {}
    for module_name in ("sglang", "sgl_kernel", "triton", "torch"):
        spec = importlib.util.find_spec(module_name)
        paths[module_name] = spec.origin if spec else None
    return paths


def projection_boundary() -> dict[str, str]:
    boundary: dict[str, str] = {}
    try:
        sample_input = torch.randn(1, HIDDEN_SIZE, dtype=DTYPE, device="cuda")
        sample_weight = torch.randn(
            HIDDEN_SIZE, HIDDEN_SIZE, dtype=DTYPE, device="cuda"
        ).transpose(0, 1)
        dsv3_fused_a_gemm(sample_input, sample_weight)
        boundary["dsv3_fused_a_gemm"] = "supported"
    except Exception as error:
        boundary["dsv3_fused_a_gemm"] = f"{type(error).__name__}: {error}"
    try:
        sample_input = torch.randn(1, HIDDEN_SIZE, dtype=DTYPE, device="cuda")
        sample_weight = torch.randn(
            HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, device="cuda"
        )
        input_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        weight_scale = torch.ones(
            HIDDEN_SIZE, dtype=torch.float32, device="cuda"
        )
        fp8_scaled_mm(
            sample_input.to(torch.float8_e4m3fn),
            sample_weight,
            input_scale,
            weight_scale,
            DTYPE,
        )
        boundary["fp8_scaled_mm"] = "supported"
    except Exception as error:
        boundary["fp8_scaled_mm"] = f"{type(error).__name__}: {error}"
    return boundary


def run_case(
    batch_size: int,
    context_length: int,
    norm_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> dict[str, object]:
    device = torch.device("cuda")
    allocated_before_case = int(torch.cuda.memory_allocated(device))
    blocks_per_sequence = context_length // BLOCK_SIZE
    num_blocks = batch_size * blocks_per_sequence
    key_cache = torch.zeros(
        num_blocks,
        BLOCK_SIZE,
        ATTENTION_HEADS,
        HEAD_SIZE,
        dtype=DTYPE,
        device=device,
    )
    value_cache = torch.zeros_like(key_cache)
    input_state = torch.randn(batch_size, HIDDEN_SIZE, dtype=DTYPE, device=device) * 0.1
    residual_state = torch.randn_like(input_state) * 0.1
    slot_mapping = (
        torch.arange(batch_size, dtype=torch.int64, device=device) * context_length
    )

    cache_bytes = key_cache.numel() * key_cache.element_size() * 2
    state_bytes = input_state.numel() * input_state.element_size() * 2
    slot_bytes = slot_mapping.numel() * slot_mapping.element_size()
    slot_allocator_bytes = allocator_block_bytes(slot_bytes)
    qkv_bytes = batch_size * 3 * ATTENTION_DIM * input_state.element_size()
    gate_bytes = batch_size * 2 * INTERMEDIATE_SIZE * input_state.element_size()
    activated_bytes = batch_size * INTERMEDIATE_SIZE * input_state.element_size()
    down_bytes = input_state.numel() * input_state.element_size()
    norm_transient_bytes = 2 * input_state.numel() * input_state.element_size()
    output_bytes = qkv_bytes + gate_bytes + activated_bytes + down_bytes
    reference_clone_bytes = 2 * input_state.numel() * input_state.element_size()
    weight_bytes = (
        norm_weight.numel() * norm_weight.element_size()
        + qkv_weight.numel() * qkv_weight.element_size()
        + gate_weight.numel() * gate_weight.element_size()
        + down_weight.numel() * down_weight.element_size()
    )
    predicted_setup_case_bytes = cache_bytes + state_bytes + slot_allocator_bytes
    predicted_peak_case_bytes = cache_bytes + max(
        state_bytes + slot_allocator_bytes + reference_clone_bytes + norm_transient_bytes,
        state_bytes + slot_allocator_bytes + reference_clone_bytes + output_bytes,
    )

    norm_sample_input = input_state.clone()
    norm_sample_residual = residual_state.clone()
    fused_add_rmsnorm(norm_sample_input, norm_sample_residual, norm_weight, 1e-6)
    qkv_sample = F.linear(norm_sample_input, qkv_weight)
    key_sample = qkv_sample[:, ATTENTION_DIM : 2 * ATTENTION_DIM].reshape(
        batch_size, ATTENTION_HEADS, HEAD_SIZE
    )
    value_sample = qkv_sample[:, 2 * ATTENTION_DIM :].reshape(
        batch_size, ATTENTION_HEADS, HEAD_SIZE
    )
    gate_sample = F.linear(norm_sample_input, gate_weight)
    activated_sample = silu_and_mul(gate_sample)

    per_operator_ms = {
        "fused_add_rmsnorm": measure_cuda_operation(
            lambda: fused_add_rmsnorm(
                norm_sample_input,
                norm_sample_residual,
                norm_weight,
                1e-6,
            ),
            WARMUP_STEPS,
            TIMED_STEPS,
        ),
        "qkv_projection_control": measure_cuda_operation(
            lambda: F.linear(norm_sample_input, qkv_weight),
            WARMUP_STEPS,
            TIMED_STEPS,
        ),
        "kv_cache_write": measure_cuda_operation(
            lambda: reshape_and_cache_flash(
                key_sample,
                value_sample,
                key_cache,
                value_cache,
                slot_mapping,
            ),
            WARMUP_STEPS,
            TIMED_STEPS,
        ),
        "gate_projection_control": measure_cuda_operation(
            lambda: F.linear(norm_sample_input, gate_weight),
            WARMUP_STEPS,
            TIMED_STEPS,
        ),
        "silu_and_mul": measure_cuda_operation(
            lambda: silu_and_mul(gate_sample),
            WARMUP_STEPS,
            TIMED_STEPS,
        ),
        "down_projection_control": measure_cuda_operation(
            lambda: F.linear(activated_sample, down_weight),
            WARMUP_STEPS,
            TIMED_STEPS,
        ),
    }
    del (
        norm_sample_input,
        norm_sample_residual,
        qkv_sample,
        key_sample,
        value_sample,
        gate_sample,
        activated_sample,
    )

    allocated_after_setup = int(torch.cuda.memory_allocated(device))
    measured_setup_case_bytes = allocated_after_setup - allocated_before_case
    if measured_setup_case_bytes != predicted_setup_case_bytes:
        raise RuntimeError(
            f"setup allocation mismatch for batch={batch_size}, context={context_length}: "
            f"predicted={predicted_setup_case_bytes}, measured={measured_setup_case_bytes}"
        )

    chain_timings: list[float] = []
    correctness: list[dict[str, object]] = []
    for step_index in range(CHAIN_WARMUP_STEPS + CHAIN_TIMED_STEPS):
        input_pointer = input_state.data_ptr()
        residual_pointer = residual_state.data_ptr()
        input_before = input_state.clone()
        residual_before = residual_state.clone()
        step_slot_mapping = slot_mapping + step_index
        torch.cuda.reset_peak_memory_stats(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        qkv_output, gate_output, activated_output, down_output = block_forward(
            input_state,
            residual_state,
            norm_weight,
            qkv_weight,
            gate_weight,
            down_weight,
            key_cache,
            value_cache,
            step_slot_mapping,
        )
        end.record()
        torch.cuda.synchronize()
        step_elapsed_ms = begin.elapsed_time(end)
        if step_index >= CHAIN_WARMUP_STEPS:
            chain_timings.append(step_elapsed_ms)
        peak_during_forward = int(torch.cuda.max_memory_allocated(device))
        measured_peak_case_bytes = peak_during_forward - allocated_before_case
        if peak_during_forward >= MEMORY_LIMIT_BYTES:
            raise RuntimeError(
                f"48GB live-allocation limit exceeded: {peak_during_forward} bytes"
            )

        (
            residual_expected,
            normed_expected,
            qkv_expected,
            gate_expected,
            activated_expected,
        down_expected,
        ) = reference_block(
            input_before,
            residual_before,
            norm_weight,
            qkv_weight,
            gate_weight,
            down_weight,
        )
        qkv_operator_expected = F.linear(input_state, qkv_weight)
        gate_operator_expected = F.linear(input_state, gate_weight)
        activated_operator_expected = (
            F.silu(gate_output[:, :INTERMEDIATE_SIZE].float())
            * gate_output[:, INTERMEDIATE_SIZE:].float()
        ).to(DTYPE)
        down_operator_expected = F.linear(activated_output, down_weight)
        key_expected = qkv_output[:, ATTENTION_DIM : 2 * ATTENTION_DIM].reshape(
            batch_size, ATTENTION_HEADS, HEAD_SIZE
        )
        value_expected = qkv_output[:, 2 * ATTENTION_DIM :].reshape(
            batch_size, ATTENTION_HEADS, HEAD_SIZE
        )
        actual_key = key_cache.reshape(-1, ATTENTION_HEADS, HEAD_SIZE)[step_slot_mapping]
        actual_value = value_cache.reshape(-1, ATTENTION_HEADS, HEAD_SIZE)[
            step_slot_mapping
        ]
        step_result = {
            "step": step_index,
            "dtype_preserved": bool(
                input_state.dtype == DTYPE
                and residual_state.dtype == DTYPE
                and qkv_output.dtype == DTYPE
                and gate_output.dtype == DTYPE
                and activated_output.dtype == DTYPE
                and down_output.dtype == DTYPE
            ),
            "input_pointer_preserved": bool(input_state.data_ptr() == input_pointer),
            "residual_pointer_preserved": bool(
                residual_state.data_ptr() == residual_pointer
            ),
            "residual_mutation": comparison(
                residual_state, residual_expected, NUMERICAL_GATES["norm"]
            ),
            "norm_mutation": comparison(
                input_state, normed_expected, NUMERICAL_GATES["norm"]
            ),
            "qkv_projection": comparison(
                qkv_output, qkv_operator_expected, NUMERICAL_GATES["projection"]
            ),
            "gate_projection": comparison(
                gate_output, gate_operator_expected, NUMERICAL_GATES["projection"]
            ),
            "activation": comparison(
                activated_output,
                activated_operator_expected,
                NUMERICAL_GATES["activation"],
            ),
            "down_projection": comparison(
                down_output, down_operator_expected, NUMERICAL_GATES["projection"]
            ),
            "key_cache": comparison(
                actual_key, key_expected, NUMERICAL_GATES["cache"]
            ),
            "value_cache": comparison(
                actual_value, value_expected, NUMERICAL_GATES["cache"]
            ),
            "whole_chain": {
                "normalized_rmse": normalized_rmse(down_output, down_expected),
                "max_abs": float(
                    (down_output.float() - down_expected.float()).abs().max().item()
                ),
                "mean_abs": float(
                    (down_output.float() - down_expected.float()).abs().mean().item()
                ),
            },
            "peak_during_forward_bytes": peak_during_forward,
            "measured_peak_case_bytes": measured_peak_case_bytes,
        }
        correctness.append(step_result)
        input_state = down_output.detach()
        del (
            input_before,
            residual_before,
            step_slot_mapping,
            qkv_output,
            gate_output,
            activated_output,
            down_output,
            residual_expected,
            normed_expected,
            qkv_expected,
            gate_expected,
            activated_expected,
            down_expected,
            key_expected,
            value_expected,
            qkv_operator_expected,
            gate_operator_expected,
            activated_operator_expected,
            down_operator_expected,
            actual_key,
            actual_value,
        )

    failed_checks = [
        f"step={step['step']}, check={check}"
        for step in correctness
        for check, value in step.items()
        if isinstance(value, dict)
        and (
            "allclose" in value
            and not value["allclose"]
            or check == "whole_chain"
            and value["normalized_rmse"]
            > NUMERICAL_GATES["whole_chain"]["normalized_rmse_max"]
        )
    ]
    if failed_checks:
        raise RuntimeError("numerical gate failures: " + "; ".join(failed_checks))
    if not all(step["dtype_preserved"] for step in correctness):
        raise RuntimeError("dtype preservation failure")
    if not all(
        step["input_pointer_preserved"] and step["residual_pointer_preserved"]
        for step in correctness
    ):
        raise RuntimeError("in-place mutation contract failure")

    result = {
        "batch_size": batch_size,
        "context_length": context_length,
        "timed_steps": CHAIN_TIMED_STEPS,
        "chain_warmup_steps": CHAIN_WARMUP_STEPS,
        "warmup_steps_per_operator": WARMUP_STEPS,
        "per_operator_ms": per_operator_ms,
        "whole_chain_ms": timing_stats(chain_timings),
        "correctness": correctness,
        "memory": {
            "weight_bytes": weight_bytes,
            "allocated_before_case_bytes": allocated_before_case,
            "kv_workspace_bytes": cache_bytes,
            "tensor_workspace_bytes": state_bytes
            + slot_bytes
            + norm_transient_bytes
            + output_bytes,
            "slot_allocator_bytes": slot_allocator_bytes,
            "predicted_setup_case_bytes": predicted_setup_case_bytes,
            "measured_setup_live_bytes": allocated_after_setup,
            "measured_setup_case_bytes": measured_setup_case_bytes,
            "predicted_peak_case_bytes": predicted_peak_case_bytes,
            "measured_peak_live_bytes": max(
                step["peak_during_forward_bytes"] for step in correctness
            ),
            "measured_peak_case_bytes": max(
                step["measured_peak_case_bytes"] for step in correctness
            ),
            "limit_bytes": MEMORY_LIMIT_BYTES,
        },
    }
    if abs(predicted_peak_case_bytes - result["memory"]["measured_peak_case_bytes"]) > 1024:
        raise RuntimeError(
            f"peak allocation mismatch for batch={batch_size}, context={context_length}: "
            f"predicted={predicted_peak_case_bytes}, measured={result['memory']['measured_peak_case_bytes']}"
        )
    del input_state, residual_state, key_cache, value_cache, slot_mapping
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("mi300-block-results.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    torch.manual_seed(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one GPU, found {torch.cuda.device_count()}")
    device_properties = torch.cuda.get_device_properties(0)
    if "gfx942" not in device_properties.gcnArchName:
        raise RuntimeError(f"expected gfx942, found {device_properties.gcnArchName}")

    norm_weight = torch.ones(HIDDEN_SIZE, dtype=DTYPE, device="cuda")
    qkv_weight = (
        torch.randn(3 * ATTENTION_DIM, HIDDEN_SIZE, dtype=DTYPE, device="cuda") * 0.02
    )
    gate_weight = (
        torch.randn(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=DTYPE, device="cuda")
        * 0.02
    )
    down_weight = (
        torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=DTYPE, device="cuda") * 0.02
    )
    weight_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (norm_weight, qkv_weight, gate_weight, down_weight)
    )
    if weight_bytes >= WEIGHT_LIMIT_BYTES:
        raise RuntimeError(f"synthetic weights exceed 4GB: {weight_bytes} bytes")

    warmup_input = torch.randn(1, HIDDEN_SIZE, dtype=DTYPE, device="cuda")
    warmup_residual = torch.randn_like(warmup_input)
    fused_add_rmsnorm(warmup_input, warmup_residual, norm_weight, 1e-6)
    warmup_gate = torch.randn(
        1, 2 * INTERMEDIATE_SIZE, dtype=DTYPE, device="cuda"
    )
    silu_and_mul(warmup_gate)
    warmup_key_cache = torch.zeros(
        1,
        BLOCK_SIZE,
        ATTENTION_HEADS,
        HEAD_SIZE,
        dtype=DTYPE,
        device="cuda",
    )
    warmup_value_cache = torch.zeros_like(warmup_key_cache)
    warmup_key = torch.randn(
        1, ATTENTION_HEADS, HEAD_SIZE, dtype=DTYPE, device="cuda"
    )
    warmup_value = torch.randn_like(warmup_key)
    warmup_slots = torch.zeros(1, dtype=torch.int64, device="cuda")
    reshape_and_cache_flash(
        warmup_key,
        warmup_value,
        warmup_key_cache,
        warmup_value_cache,
        warmup_slots,
    )
    warmup_qkv_output = F.linear(warmup_input, qkv_weight)
    warmup_gate_output = F.linear(warmup_input, gate_weight)
    warmup_activated = torch.randn(
        1, INTERMEDIATE_SIZE, dtype=DTYPE, device="cuda"
    )
    warmup_down_output = F.linear(warmup_activated, down_weight)
    del (
        warmup_input,
        warmup_residual,
        warmup_gate,
        warmup_key_cache,
        warmup_value_cache,
        warmup_key,
        warmup_value,
        warmup_slots,
        warmup_qkv_output,
        warmup_gate_output,
        warmup_activated,
        warmup_down_output,
    )
    torch.cuda.synchronize()

    results = {
        "label": "persistent mirror checkout real-GPU reduced-block benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": "repo-e2e-20260909",
        "job_id": os.environ.get("JOB_ID"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": "/opt/venv/bin/python",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "operator_paths": operator_paths(),
        "gpu": {
            "name": device_properties.name,
            "gcn_arch": device_properties.gcnArchName,
            "total_bytes": device_properties.total_memory,
            "device_count": torch.cuda.device_count(),
        },
        "image_identity": {
            "expected_image": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
            "expected_local_image_id": "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
            "source": "task specification; container hostname is not image identity",
        },
        "dimensions": {
            "hidden_size": HIDDEN_SIZE,
            "attention_heads": ATTENTION_HEADS,
            "head_size": HEAD_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "dtype": str(DTYPE),
            "block_size": BLOCK_SIZE,
        },
        "cases": [list(case) for case in CASES],
        "numerical_gates": NUMERICAL_GATES,
        "timing_method": {
            "event": "torch.cuda.Event(enable_timing=True)",
            "warmup_steps_per_operator": WARMUP_STEPS,
            "timed_steps": TIMED_STEPS,
            "statistics": ["median", "p95", "mean", "min", "max"],
        },
        "projection_control": {
            "reason": "public SGLang bf16 GEMM wrappers are unavailable in the installed HIP wheel",
            "boundary": projection_boundary(),
            "control": "torch.nn.functional.linear bf16 projection",
        },
        "synthetic_weight_bytes": weight_bytes,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "case_results": [],
    }

    for batch_size, context_length in CASES:
        results["case_results"].append(
            run_case(
                batch_size,
                context_length,
                norm_weight,
                qkv_weight,
                gate_weight,
                down_weight,
            )
        )

    results["elapsed_s"] = time.perf_counter() - started
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
