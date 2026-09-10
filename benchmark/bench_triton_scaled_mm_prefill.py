import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from sglang.kernels.ops.quantization.fp8_kernel import triton_scaled_mm


DEFAULT_CASES = (
    "int8:128:2048:2048,"
    "int8:1024:2048:2048,"
    "int8:8192:2048:2048,"
    "fp8:128:2048:2048,"
    "fp8:1024:2048:2048,"
    "fp8:8192:2048:2048"
)

NUMERICAL_GATES = {
    "int8": {"rtol": 0.15, "atol": 0.10},
    "fp8": {"rtol": 0.25, "atol": 0.15},
}


def _parse_cases(raw_cases):
    cases = []
    for raw_case in raw_cases.split(","):
        dtype_name, m, k, n = raw_case.split(":")
        if dtype_name not in NUMERICAL_GATES:
            raise ValueError(f"Unsupported dtype {dtype_name!r}; expected int8 or fp8")
        cases.append(
            {
                "dtype": dtype_name,
                "M": int(m),
                "K": int(k),
                "N": int(n),
            }
        )
    if not 1 <= len(cases) <= 6:
        raise ValueError("Expected between 1 and 6 cases")
    return cases


def _make_inputs(dtype_name, m, k, n, device):
    if dtype_name == "int8":
        input_tensor = torch.randint(-8, 9, (m, k), device=device, dtype=torch.int8)
        weight = torch.randint(-8, 9, (k, n), device=device, dtype=torch.int8)
        scale_a = 0.01 + 0.005 * torch.rand((m, 1), device=device, dtype=torch.float32)
        scale_b = 0.01 + 0.005 * torch.rand((n, 1), device=device, dtype=torch.float32)
    else:
        input_tensor = torch.clamp(
            0.1 * torch.randn((m, k), device=device, dtype=torch.float16),
            -0.3,
            0.3,
        ).to(torch.float8_e4m3fn)
        weight = torch.clamp(
            0.1 * torch.randn((k, n), device=device, dtype=torch.float16),
            -0.3,
            0.3,
        ).to(torch.float8_e4m3fn)
        scale_a = 0.1 + 0.05 * torch.rand((m, 1), device=device, dtype=torch.float32)
        scale_b = 0.1 + 0.05 * torch.rand((n, 1), device=device, dtype=torch.float32)
    return input_tensor, weight, scale_a, scale_b


def _independent_reference(input_tensor, weight, scale_a, scale_b):
    return (input_tensor.to(torch.float32) * scale_a) @ (
        weight.to(torch.float32) * scale_b.t()
    )


def _measure_warm(fn, warmup_iterations, measured_iterations):
    for _ in range(warmup_iterations):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(measured_iterations):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 1000.0


def _run_case(case, device, seed, seen_dtypes):
    dtype_name = case["dtype"]
    m, k, n = case["M"], case["K"], case["N"]
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(seed)
    input_tensor, weight, scale_a, scale_b = _make_inputs(
        dtype_name, m, k, n, device
    )
    reference = _independent_reference(input_tensor, weight, scale_a, scale_b)

    def forward():
        return triton_scaled_mm(
            input_tensor,
            weight,
            scale_a,
            scale_b,
            torch.bfloat16,
        )

    cold = dtype_name not in seen_dtypes
    cold_start = time.perf_counter()
    output = forward()
    torch.cuda.synchronize()
    cold_elapsed = time.perf_counter() - cold_start
    seen_dtypes.add(dtype_name)

    error = (output.float() - reference).abs()
    denominator = reference.abs().clamp_min(1e-6)
    gate = NUMERICAL_GATES[dtype_name]
    gate_passed = torch.allclose(
        output.float(),
        reference,
        rtol=gate["rtol"],
        atol=gate["atol"],
    )

    if m >= 4096:
        warmup_iterations = 10
        measured_iterations = 30
    else:
        warmup_iterations = 10
        measured_iterations = 100
    warm_elapsed = _measure_warm(forward, warmup_iterations, measured_iterations)
    flops = 2 * m * k * n * measured_iterations

    result = {
        **case,
        "cold": cold,
        "cold_first_forward_elapsed_seconds": cold_elapsed,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "warm_elapsed_seconds": warm_elapsed,
        "warm_throughput_tflops": flops / warm_elapsed / 1e12,
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "max_relative_error": float((error / denominator).max().item()),
        "mean_relative_error": float((error / denominator).mean().item()),
        "accuracy_gate": gate,
        "gate_passed": bool(gate_passed),
        "weight_bytes": weight.numel() * weight.element_size(),
        "activation_bytes": input_tensor.numel() * input_tensor.element_size(),
        "reference_bytes": reference.numel() * reference.element_size(),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }

    del output, reference, error, denominator, input_tensor, weight, scale_a, scale_b
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Bounded INT8/FP8 dense-linear prefill throughput study"
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, default=15194)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/HIP GPU is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    cases = _parse_cases(args.cases)
    seen_dtypes = set()
    results = []
    for case in cases:
        results.append(_run_case(case, device, args.seed, seen_dtypes))

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_architecture": list(torch.cuda.get_device_capability(device)),
        "torch_version": str(torch.__version__),
        "hip_version": str(torch.version.hip),
        "kernel": "sglang.kernels.ops.quantization.fp8_kernel.triton_scaled_mm",
        "output_dtype": "bfloat16",
        "scale_dtype": "float32",
        "reference": "independent FP32 dequantized matmul",
        "case_count": len(cases),
        "cases": results,
        "limits": {
            "max_cases": 6,
            "weight_limit_bytes": 4 * 1024**3,
            "total_live_allocation_limit_bytes": 48 * 1024**3,
            "wall_limit_seconds": 7200,
        },
    }

    serialized = json.dumps(record, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
