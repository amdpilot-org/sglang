#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiter
import torch
from aiter.jit.utils.chip_info import get_gfx_runtime
from sglang.kernels.ops.quantization.fp8_kernel import triton_scaled_mm


CASES = [
    {"M": 1, "N": 2048, "K": 2048},
    {"M": 64, "N": 2048, "K": 2048},
    {"M": 1024, "N": 2048, "K": 2048},
    {"M": 4096, "N": 2048, "K": 2048},
]

AITER_GATE = {"rtol": 1e-2, "atol": 1e-2}
SGLANG_GATE = {"rtol": 0.15, "atol": 0.1}
WARMUP_ITERATIONS = 3
MEASURE_ITERATIONS = 30


def git_head(path):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def timing_summary(values):
    ordered = sorted(values)
    mean = statistics.fmean(values)
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "coefficient_of_variation": statistics.pstdev(values) / mean,
        "raw_ms": values,
    }


def measure_complete_block(function):
    for _ in range(WARMUP_ITERATIONS):
        function()
    torch.cuda.synchronize()

    timings = []
    for _ in range(MEASURE_ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return timing_summary(timings)


def profile_native_kernels(function):
    function()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        function()
        torch.cuda.synchronize()

    kernels = []
    for event in profiler.key_averages():
        if event.self_device_time_total > 0:
            kernels.append(
                {
                    "name": event.key,
                    "self_device_time_us": event.self_device_time_total,
                }
            )
    return kernels


def output_difference(output, reference):
    difference = (output.float() - reference).abs()
    relative = difference / reference.abs().clamp_min(1e-6)
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "max_relative_error": relative.max().item(),
        "finite": bool(torch.isfinite(output).all().item()),
    }


def run_case(case, seed):
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    M, N, K = case["M"], case["N"], case["K"]
    x = torch.randint(-8, 9, (M, K), dtype=torch.int8, device="cuda")
    weight = torch.randint(-8, 9, (N, K), dtype=torch.int8, device="cuda")
    weight_t = weight.T.contiguous()
    x_scale = (0.05 + 0.05 * torch.rand((M, 1), device="cuda")).float()
    weight_scale = (0.05 + 0.05 * torch.rand((N, 1), device="cuda")).float()

    x_dequantized = x.float() * x_scale
    weight_dequantized = weight.float() * weight_scale
    reference = torch.matmul(x_dequantized, weight_dequantized.T)

    aiter_output = aiter.gemm_a8w8_CK(
        x,
        weight,
        x_scale,
        weight_scale,
        bias=None,
        dtype=torch.float16,
    )
    triton_output = triton_scaled_mm(
        x,
        weight_t,
        x_scale,
        weight_scale,
        torch.float16,
    )

    torch.testing.assert_close(
        aiter_output.float(),
        reference,
        rtol=AITER_GATE["rtol"],
        atol=AITER_GATE["atol"],
    )
    torch.testing.assert_close(
        triton_output.float(),
        reference,
        rtol=SGLANG_GATE["rtol"],
        atol=SGLANG_GATE["atol"],
    )

    assert aiter_output.data_ptr() != triton_output.data_ptr()
    assert aiter_output.data_ptr() not in {x.data_ptr(), weight.data_ptr()}
    assert triton_output.data_ptr() not in {x.data_ptr(), weight.data_ptr()}

    aiter_timing = measure_complete_block(
        lambda: aiter.gemm_a8w8_CK(
            x,
            weight,
            x_scale,
            weight_scale,
            bias=None,
            dtype=torch.float16,
        )
    )
    triton_timing = measure_complete_block(
        lambda: triton_scaled_mm(
            x,
            weight_t,
            x_scale,
            weight_scale,
            torch.float16,
        )
    )

    aiter_kernels = profile_native_kernels(
        lambda: aiter.gemm_a8w8_CK(
            x,
            weight,
            x_scale,
            weight_scale,
            bias=None,
            dtype=torch.float16,
        )
    )
    triton_kernels = profile_native_kernels(
        lambda: triton_scaled_mm(
            x,
            weight_t,
            x_scale,
            weight_scale,
            torch.float16,
        )
    )

    return {
        "dimensions": case,
        "seed": seed,
        "input_dtype": str(torch.int8),
        "scale_dtype": str(torch.float32),
        "output_dtype": str(torch.float16),
        "weight_bytes": weight.numel() * weight.element_size(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "current_allocated_bytes": torch.cuda.memory_allocated(),
        "input_pointer": x.data_ptr(),
        "weight_pointer": weight.data_ptr(),
        "triton_weight_pointer": weight_t.data_ptr(),
        "aiter_output_pointer": aiter_output.data_ptr(),
        "triton_output_pointer": triton_output.data_ptr(),
        "aiter": {
            "backend": "AITER CK",
            "dispatch": "aiter.gemm_a8w8_CK",
            "gate": AITER_GATE,
            "difference": output_difference(aiter_output, reference),
            "timing": aiter_timing,
            "profiled_kernels": aiter_kernels,
        },
        "triton": {
            "backend": "SGLang Triton",
            "dispatch": "sglang.kernels.ops.quantization.fp8_kernel.triton_scaled_mm",
            "gate": SGLANG_GATE,
            "difference": output_difference(triton_output, reference),
            "timing": triton_timing,
            "profiled_kernels": triton_kernels,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    started = time.monotonic()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one GPU, found {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is unavailable")

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    architecture = get_gfx_runtime()
    if architecture != "gfx942":
        raise RuntimeError(f"Expected gfx942, found {architecture}")

    aiter_native = Path(aiter.__file__).parent / "jit" / "module_gemm_a8w8.so"
    if not aiter_native.exists():
        raise RuntimeError(f"AITER native module missing: {aiter_native}")

    triton_cache = Path(os.environ.get("TRITON_CACHE_DIR", ""))
    triton_native_paths = []
    if triton_cache.exists():
        triton_native_paths = [
            str(path) for path in triton_cache.rglob("*.hsaco")
        ]

    results = []
    for index, case in enumerate(CASES):
        results.append(run_case(case, seed=15194 + index))

    elapsed_seconds = time.monotonic() - started
    report = {
        "label": "bounded INT8 dense-linear backend comparison",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "gpu": {
            "name": gpu_name,
            "architecture": architecture,
            "torch_capability": list(capability),
            "device_count": torch.cuda.device_count(),
        },
        "image": {
            "reference": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
            "hostname": os.uname().nodename,
        },
        "source": {
            "sglang_commit": git_head(Path(__file__).parents[2]),
            "aiter_commit": git_head(Path(aiter.__file__).parents[1]),
            "sglang_python": str(Path(__file__).parents[2] / "python"),
            "aiter_python": str(Path(aiter.__file__).parent),
        },
        "native_paths": {
            "aiter_module": str(aiter_native),
            "sglang_triton_source": triton_scaled_mm.__code__.co_filename,
            "triton_cache": str(triton_cache),
            "triton_hsacos": triton_native_paths,
        },
        "method": {
            "reference": "independent FP32 dequantized matmul of identical synthetic INT8 operands and FP32 scales",
            "timing": f"{WARMUP_ITERATIONS} warmups and {MEASURE_ITERATIONS} CUDA-event measurements; complete call includes output allocation",
            "workload_case_count": len(CASES),
            "weight_limit_bytes": 4 * 1024**3,
            "wall_limit_seconds": 7200,
        },
        "elapsed_seconds": elapsed_seconds,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
