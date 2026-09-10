import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from sglang.kernels.ops.quantization.int8_kernel import w8a8_block_int8_matmul


N = 1024
K = 1024
BLOCK = 128
OUTPUT_DTYPE = torch.bfloat16
INPUT_DISTRIBUTIONS = ("zero_tiny", "mixed_magnitude", "cancellation_skewed")
MS = (1, 4096)


def make_weight(device):
    torch.manual_seed(15194)
    weight = torch.randint(-127, 128, (N, K), device=device, dtype=torch.int8)
    weight[:, :BLOCK] = 1
    scales = torch.logspace(
        -6,
        -2,
        steps=(N // BLOCK) * (K // BLOCK),
        device=device,
        dtype=torch.float32,
    ).reshape(N // BLOCK, K // BLOCK)
    return weight, scales


def make_activation(name, m, device):
    if name == "zero_tiny":
        activation = torch.zeros((m, K), device=device, dtype=torch.int8)
        tiny = torch.rand((m, K), device=device) < 0.1
        activation[tiny] = torch.where(
            torch.rand((m, K), device=device)[tiny] < 0.5, -1, 1
        ).to(torch.int8)
        scales = torch.full((m, K // BLOCK), 1e-6, device=device)
    elif name == "mixed_magnitude":
        activation = torch.randint(-127, 128, (m, K), device=device, dtype=torch.int8)
        scales = torch.logspace(
            -6, -2, steps=m * (K // BLOCK), device=device, dtype=torch.float32
        ).reshape(m, K // BLOCK)
    elif name == "cancellation_skewed":
        activation = torch.zeros((m, K), device=device, dtype=torch.int8)
        alternating = torch.arange(m, device=device) % 4 == 0
        activation[alternating, 0::2] = 127
        activation[alternating, 1::2] = -127
        skewed = ~alternating
        nonzero = torch.rand((m, K), device=device) < 0.1
        signs = torch.where(torch.rand((m, K), device=device) < 0.5, -127, 127).to(
            torch.int8
        )
        activation[skewed] = torch.where(nonzero, signs, torch.zeros_like(signs))[
            skewed
        ]
        scales = torch.full((m, K // BLOCK), 1e-3, device=device)
    else:
        raise ValueError(f"unknown input distribution: {name}")
    return activation, scales


def dequantized_reference(activation, activation_scales, weight, weight_scales):
    activation_float = activation.to(torch.float32) * activation_scales.repeat_interleave(
        BLOCK, dim=1
    )
    weight_float = weight.to(torch.float32) * weight_scales.repeat_interleave(
        BLOCK, dim=0
    ).repeat_interleave(BLOCK, dim=1)
    return (activation_float @ weight_float.T).to(OUTPUT_DTYPE)


def measure(kernel, warmups=3, samples=20):
    for _ in range(warmups):
        kernel()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(samples):
        start.record()
        kernel()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return {
        "warmups": warmups,
        "samples": samples,
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.fmean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "coefficient_of_variation": statistics.pstdev(timings)
        / statistics.fmean(timings),
    }


def newest_hsaco():
    cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    candidates = [path for path in cache.rglob("*.hsaco") if path.is_file()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    weight, weight_scales = make_weight(device)
    cases = []

    for m in MS:
        for distribution in INPUT_DISTRIBUTIONS:
            activation, activation_scales = make_activation(distribution, m, device)

            def run_kernel():
                return w8a8_block_int8_matmul(
                    activation,
                    weight,
                    activation_scales,
                    weight_scales,
                    [BLOCK, BLOCK],
                    OUTPUT_DTYPE,
                )

            actual = run_kernel()
            reference = dequantized_reference(
                activation, activation_scales, weight, weight_scales
            )
            difference = (
                actual.to(torch.float32) - reference.to(torch.float32)
            ).abs()
            reference_mean = reference.to(torch.float32).abs().mean()
            relative_error = difference.mean() / reference_mean.clamp(min=1e-30)
            timing = measure(run_kernel)
            cases.append(
                {
                    "m": m,
                    "n": N,
                    "k": K,
                    "block": [BLOCK, BLOCK],
                    "input_distribution": distribution,
                    "output_dtype": str(OUTPUT_DTYPE),
                    "finite": bool(torch.isfinite(actual).all()),
                    "max_abs_error": difference.max().item(),
                    "mean_abs_error": difference.mean().item(),
                    "mean_relative_error": relative_error.item(),
                    "gate": "mean_relative_error < 0.02",
                    "gate_passed": bool(
                        torch.isfinite(actual).all() and relative_error.item() < 0.02
                    ),
                    "timing": timing,
                }
            )
            del actual, reference, difference, activation, activation_scales

    properties = torch.cuda.get_device_properties(0)
    result = {
        "label": "gfx942 structured-input INT8 block-GEMM study",
        "source_commit": os.popen("git rev-parse HEAD").read().strip(),
        "python": "/opt/venv/bin/python",
        "torch": torch.__version__,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "gcn_architecture": properties.gcnArchName,
            "multi_processor_count": properties.multi_processor_count,
        },
        "kernel": "sglang.kernels.ops.quantization.int8_kernel.w8a8_block_int8_matmul",
        "source_path": str(
            Path(
                __import__(
                    "sglang.kernels.ops.quantization.int8_kernel", fromlist=[""]
                ).__file__
            )
        ),
        "native_path": newest_hsaco(),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "weight_bytes": weight.numel(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "reference": "independent float32 dequantized matmul, cast to bfloat16",
        "cases": cases,
        "all_gates_passed": all(case["gate_passed"] for case in cases),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
