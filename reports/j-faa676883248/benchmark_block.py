#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.kernels.ops.activation import _SILU_AND_MUL
from sglang.kernels.ops.attention.dsv4 import linear_bf16_fp32
from sglang.kernels.ops.layernorm.norm import rmsnorm as jit_rmsnorm
from sglang.kernels.spec import KernelBackend


CASES = [
    {"name": "decode", "tokens": 1},
    {"name": "small_batch", "tokens": 64},
    {"name": "prefill", "tokens": 1024},
    {"name": "large_prefill", "tokens": 4096},
]
CONFIGS = [
    {"name": "jit_norm_aot_act", "activation": KernelBackend.AOT},
    {"name": "jit_norm_aiter_act", "activation": KernelBackend.AITER},
    {"name": "jit_norm_torch_act", "activation": KernelBackend.TORCH},
]
HIDDEN = 8192
INTERMEDIATE = 4096
OUTPUT = 4096
WARMUP = 10
ITERATIONS = 30
REPEATS = 5
EPS = 1e-6


def tensor_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def error(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs()
    return {
        "max_abs": float(diff.max().item()),
        "max_rel": float((diff / denom.clamp_min(1e-6)).max().item()),
    }


def passes(actual, expected, atol, rtol):
    return bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))


def time_call(call, warmup, iterations):
    for _ in range(warmup):
        call()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def summarize(values):
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
        "iqr_ms": statistics.quantiles(values, n=4)[2]
        - statistics.quantiles(values, n=4)[0],
        "raw_ms": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required")

    torch.manual_seed(29630)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    norm_weight = torch.randn(HIDDEN, device=device, dtype=dtype)
    projection_weight = torch.randn(OUTPUT, INTERMEDIATE, device=device, dtype=dtype)
    weight_bytes = tensor_bytes(norm_weight, projection_weight)
    if weight_bytes >= 4 * 1024**3:
        raise RuntimeError("weights exceed the 4 GiB limit")

    results = {
        "schema": "sglang-reduced-block-v1",
        "source_commit": "0084030179bfba86bfeb6d43f7997d4076329d2c",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "block": {
            "input": [None, HIDDEN],
            "activation_input": [None, HIDDEN],
            "activation_output": [None, INTERMEDIATE],
            "projection_output": [None, OUTPUT],
            "input_dtype": "torch.bfloat16",
            "projection_output_dtype": "torch.float32",
            "operators": [
                "sglang.kernels.ops.layernorm.norm.rmsnorm",
                "sglang.kernels.ops.activation._SILU_AND_MUL.forward",
                "sglang.kernels.ops.attention.dsv4.linear_bf16_fp32",
            ],
        },
        "limits": {
            "weight_limit_bytes": 4 * 1024**3,
            "live_allocation_limit_bytes": 48 * 1024**3,
            "workload_cases": len(CASES),
            "configurations": len(CONFIGS),
            "warmup_iterations_per_measurement": WARMUP,
            "measured_iterations_per_repeat": ITERATIONS,
            "repeats": REPEATS,
        },
        "accuracy_gates": {
            "norm": {"atol": 5e-2, "rtol": 2e-2},
            "activation": {"atol": 5e-2, "rtol": 2e-2},
            "projection": {"atol": 1e-1, "rtol": 8e-2},
            "chain": {"atol": 1e-1, "rtol": 8e-2},
        },
        "weight_bytes": weight_bytes,
        "cases": [],
    }

    for case in CASES:
        tokens = case["tokens"]
        torch.manual_seed(29630 + tokens)
        block_input = torch.randn(tokens, HIDDEN, device=device, dtype=dtype)
        norm_out = torch.empty_like(block_input)
        activation_out = torch.empty(
            tokens, INTERMEDIATE, device=device, dtype=dtype
        )

        norm_reference = block_input.float()
        norm_reference = norm_reference * torch.rsqrt(
            norm_reference.pow(2).mean(dim=-1, keepdim=True) + EPS
        )
        norm_reference = (norm_reference * norm_weight.float()).to(dtype)
        activation_reference = (
            F.silu(norm_reference[..., :INTERMEDIATE].float()).to(dtype)
            * norm_reference[..., INTERMEDIATE:]
        )
        projection_reference = torch.mm(
            activation_reference.float(), projection_weight.float().t()
        )

        input_before = block_input.clone()
        norm_weight_before = norm_weight.clone()
        projection_weight_before = projection_weight.clone()
        case_result = {
            "name": case["name"],
            "tokens": tokens,
            "input_bytes": tensor_bytes(block_input),
            "live_bytes": tensor_bytes(
                block_input,
                norm_weight,
                projection_weight,
                norm_out,
                activation_out,
            ),
            "configurations": [],
        }
        if case_result["live_bytes"] >= 48 * 1024**3:
            raise RuntimeError("live allocations exceed the 48 GiB limit")

        for config in CONFIGS:
            activation_backend = config["activation"]

            def norm_call():
                jit_rmsnorm(block_input, norm_weight, norm_out, EPS)
                return norm_out

            def activation_call():
                return _SILU_AND_MUL.forward(
                    norm_out, activation_out, backend=activation_backend
                )

            def projection_call():
                return linear_bf16_fp32(activation_out, projection_weight)

            def chain_call():
                jit_rmsnorm(block_input, norm_weight, norm_out, EPS)
                _SILU_AND_MUL.forward(
                    norm_out, activation_out, backend=activation_backend
                )
                return linear_bf16_fp32(activation_out, projection_weight)

            norm_result = norm_call()
            activation_result = activation_call()
            projection_result = projection_call()
            chain_result = chain_call()
            torch.cuda.synchronize()

            projection_input_reference = torch.mm(
                activation_result.float(), projection_weight.float().t()
            )

            norm_error = error(norm_result, norm_reference)
            activation_error = error(activation_result, activation_reference)
            projection_error = error(projection_result, projection_input_reference)
            chain_error = error(chain_result, projection_reference)
            norm_pass = passes(norm_result, norm_reference, 5e-2, 2e-2)
            activation_pass = passes(
                activation_result, activation_reference, 5e-2, 2e-2
            )
            projection_pass = passes(
                projection_result, projection_input_reference, 1e-1, 8e-2
            )
            chain_pass = passes(chain_result, projection_reference, 1e-1, 8e-2)
            mutation_pass = bool(
                torch.equal(block_input, input_before)
                and torch.equal(norm_weight, norm_weight_before)
                and torch.equal(projection_weight, projection_weight_before)
            )
            identity_pass = bool(
                norm_result is norm_out and activation_result is activation_out
            )
            dtype_pass = bool(
                norm_result.dtype == dtype
                and activation_result.dtype == dtype
                and projection_result.dtype == torch.float32
                and chain_result.dtype == torch.float32
            )

            timings = {}
            for label, call in [
                ("norm", norm_call),
                ("activation", activation_call),
                ("projection", projection_call),
                ("chain", chain_call),
            ]:
                timings[label] = summarize(
                    [
                        time_call(call, WARMUP, ITERATIONS)
                        for _ in range(REPEATS)
                    ]
                )

            case_result["configurations"].append(
                {
                    "name": config["name"],
                    "norm_backend": "sglang JIT RMSNorm",
                    "activation_backend": activation_backend.value,
                    "projection_backend": "cublas bf16 matmul with fp32 output",
                    "accuracy": {
                        "norm": {**norm_error, "pass": norm_pass},
                        "activation": {
                            **activation_error,
                            "pass": activation_pass,
                        },
                        "projection": {
                            **projection_error,
                            "pass": projection_pass,
                        },
                        "chain": {**chain_error, "pass": chain_pass},
                    },
                    "contracts": {
                        "input_and_weights_unchanged": mutation_pass,
                        "explicit_out_identity": identity_pass,
                        "dtype": dtype_pass,
                    },
                    "timings_ms": timings,
                }
            )

        results["cases"].append(case_result)
        del block_input, norm_out, activation_out
        del norm_reference, activation_reference, projection_reference
        del input_before, norm_weight_before, projection_weight_before
        torch.cuda.empty_cache()

    results["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
