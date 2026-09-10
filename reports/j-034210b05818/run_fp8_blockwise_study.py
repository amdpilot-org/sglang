import json
import math
import os
import statistics
import sys
import time

os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch

from sglang.kernels.ops.quantization import fp8_kernel
from sglang.srt.layers.quantization.fp8_utils import (
    aiter_w8a8_block_fp8_linear,
    triton_w8a8_block_fp8_linear,
)


FP8_MAX = 448.0
RTOL = 5e-2
ATOL = 1e-1
WARMUP_FORWARDS = 10
TIMED_FORWARDS = 30


def quantize_weight_blockwise(weight, block_size=128):
    n, k = weight.shape
    tiles = weight.float().reshape(n // block_size, block_size, k // block_size, block_size)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / FP8_MAX
    weight_fp8 = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    weight_dequant = (
        weight_fp8.float() * scale[:, None, :, None]
    ).reshape(n, k)
    return weight_fp8.reshape(n, k), scale, weight_dequant


def timing_stats(samples_ms):
    mean = statistics.fmean(samples_ms)
    stdev = statistics.stdev(samples_ms)
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": mean,
        "stdev_ms": stdev,
        "coefficient_of_variation": stdev / mean,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "approx_95pct_ci_ms": 1.96 * stdev / math.sqrt(len(samples_ms)),
    }


def run_case(name, forward, x, weight_fp8, weight_scale, reference):
    for _ in range(WARMUP_FORWARDS):
        forward(x, weight_fp8, weight_scale)
    torch.cuda.synchronize()

    samples = []
    for _ in range(TIMED_FORWARDS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward(x, weight_fp8, weight_scale)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    output = forward(x, weight_fp8, weight_scale)
    error = (output.float() - reference).abs()
    relative = error / (reference.abs() + 1e-6)
    gate_passed = bool(
        torch.allclose(output.float(), reference, rtol=RTOL, atol=ATOL)
    )
    return {
        "configuration": name,
        "gate_passed": gate_passed,
        "max_abs_error": error.max().item(),
        "max_relative_error": relative.max().item(),
        "timing": timing_stats(samples),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_fp8_blockwise_study.py OUTPUT_JSON")

    output_path = sys.argv[1]
    device = torch.device("cuda")
    torch.manual_seed(7)

    n, k = 4096, 512
    weight = torch.randn((n, k), device=device, dtype=torch.bfloat16) / 10
    weight_fp8, weight_scale, weight_dequant = quantize_weight_blockwise(weight)

    tuned_configs = {
        "triton_tuned_m1": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
        "triton_tuned_m512": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 2,
        },
        "triton_default": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        },
    }

    original_config_lookup = fp8_kernel.get_w8a8_block_fp8_configs
    results = []
    started = time.time()

    try:
        for m in (1, 64, 512, 2048):
            x = torch.randn((m, k), device=device, dtype=torch.bfloat16) / 10
            reference = x.float() @ weight_dequant.T

            case_results = []
            def aiter_forward(x, weight, scale):
                return aiter_w8a8_block_fp8_linear(
                    x, weight, [128, 128], scale
                )

            case_results.append(
                run_case(
                    "aiter_blockwise_fp8",
                    aiter_forward,
                    x,
                    weight_fp8,
                    weight_scale,
                    reference,
                )
            )

            for name, config in tuned_configs.items():
                fp8_kernel.get_w8a8_block_fp8_configs = lambda *_args, **_kwargs: {
                    m: config
                }
                def triton_forward(x, weight, scale):
                    return triton_w8a8_block_fp8_linear(
                        x, weight, [128, 128], scale
                    )

                case_results.append(
                    run_case(
                        name,
                        triton_forward,
                        x,
                        weight_fp8,
                        weight_scale,
                        reference,
                    )
                )

            results.append(
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "input_dtype": str(x.dtype),
                    "weight_dtype": str(weight_fp8.dtype),
                    "weight_scale_dtype": str(weight_scale.dtype),
                    "output_dtype": str(torch.bfloat16),
                    "cases": case_results,
                }
            )
    finally:
        fp8_kernel.get_w8a8_block_fp8_configs = original_config_lookup

    properties = torch.cuda.get_device_properties(0)
    report = {
        "campaign": "repo-e2e-20260909",
        "gpu": {
            "name": properties.name,
            "architecture": properties.gcnArchName,
            "uuid": str(properties.uuid),
            "total_memory_bytes": properties.total_memory,
        },
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "shape": {"n": n, "k": k},
        "accuracy_gate": {"rtol": RTOL, "atol": ATOL},
        "timing_method": {
            "warmup_forwards": WARMUP_FORWARDS,
            "timed_forwards": TIMED_FORWARDS,
            "method": "CUDA events around each real dense-linear forward",
        },
        "limits": {
            "max_workload_cases": 6,
            "max_weight_bytes": 4 * 1024**3,
            "max_live_allocations_bytes": 48 * 1024**3,
        },
        "allocated_bytes": torch.cuda.memory_allocated(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }

    with open(output_path, "w") as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
