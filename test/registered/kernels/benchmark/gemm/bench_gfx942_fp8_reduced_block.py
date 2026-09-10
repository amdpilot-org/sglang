import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from sglang.srt.models.torch_native_llama import LlamaMLP


CASES = [
    {"name": "m1", "m": 1},
    {"name": "m2", "m": 2},
    {"name": "m4", "m": 4},
    {"name": "m8", "m": 8},
    {"name": "m16", "m": 16},
    {"name": "m32", "m": 32},
]

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 512
STEPS = 4
WARMUP_SEQUENCES = 3
TIMED_SEQUENCES = 10
ABS_ERROR_LIMIT = 1.0e-4
REL_ERROR_LIMIT = 0.02
MAX_WEIGHT_BYTES = 4 * 1024**3
MAX_LIVE_BYTES = 48 * 1024**3


class Fp8Linear(nn.Module):
    def __init__(self, in_features, out_features, seed):
        super().__init__()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        weight = (
            torch.randn(
                out_features,
                in_features,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
            .clamp(-1.0, 1.0)
            .to(torch.float8_e4m3fnuz)
        )
        weight_scale = (
            torch.rand(
                out_features,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
            * 0.001
            + 0.0001
        )
        bias = (
            torch.randn(
                out_features,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            * 0.01
        )
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        scale = (
            x.abs()
            .amax(dim=1, keepdim=True)
            .to(torch.float32)
            .clamp_min(1.0e-12)
            / 240.0
        )
        qinput = (
            (x.to(torch.float32) / scale)
            .clamp(-240.0, 240.0)
            .to(torch.float8_e4m3fnuz)
        )
        return torch._scaled_mm(
            qinput,
            self.weight.t(),
            scale_a=scale,
            scale_b=self.weight_scale.view(1, -1),
            out_dtype=torch.bfloat16,
            bias=self.bias,
        )

    def dequantized_weight(self):
        return (
            self.weight.to(torch.float32) * self.weight_scale.view(-1, 1)
        ).to(torch.bfloat16)


class ReferenceLinear(nn.Module):
    def __init__(self, fp8_linear):
        super().__init__()
        self.weight = nn.Parameter(fp8_linear.dequantized_weight(), requires_grad=False)
        self.bias = nn.Parameter(fp8_linear.bias.clone(), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(
            x.to(torch.float32),
            self.weight.to(torch.float32),
            self.bias.to(torch.float32),
        ).to(torch.bfloat16)


def build_blocks():
    fp8_block = (
        LlamaMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, "silu")
        .to(torch.bfloat16)
        .cuda()
    )
    reference_block = (
        LlamaMLP(HIDDEN_SIZE, INTERMEDIATE_SIZE, "silu")
        .to(torch.bfloat16)
        .cuda()
    )
    fp8_block.gate_up_proj = Fp8Linear(HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE, 101).cuda()
    fp8_block.down_proj = Fp8Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, 102).cuda()
    reference_block.gate_up_proj = ReferenceLinear(fp8_block.gate_up_proj).cuda()
    reference_block.down_proj = ReferenceLinear(fp8_block.down_proj).cuda()
    return fp8_block, reference_block


def error_metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1.0e-5)
    return float(absolute.max().item()), float(relative.max().item())


def run_sequence(fp8_block, reference_block, initial_state, timed):
    state = initial_state
    step_results = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for step in range(STEPS):
        if timed:
            start.record()
        actual = fp8_block(state)
        if timed:
            end.record()
            end.synchronize()
        expected = reference_block(state)
        torch.cuda.synchronize()
        max_abs, max_rel = error_metrics(actual, expected)
        finite = bool(torch.isfinite(actual).all().item())
        if not finite or max_abs > ABS_ERROR_LIMIT or max_rel > REL_ERROR_LIMIT:
            raise RuntimeError(
                f"accuracy gate failed at step {step}: finite={finite}, "
                f"max_abs={max_abs}, max_rel={max_rel}"
            )
        elapsed_us = start.elapsed_time(end) * 1000.0 if timed else None
        step_results.append(
            {
                "step": step,
                "max_abs_error": max_abs,
                "max_relative_error": max_rel,
                "finite": finite,
                "latency_us": elapsed_us,
            }
        )
        state = actual
    return step_results


def percentile(values, percentile_value):
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def benchmark_case(case, fp8_block, reference_block):
    torch.manual_seed(20260910)
    initial_state = torch.randn(
        case["m"], HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    for _ in range(WARMUP_SEQUENCES):
        run_sequence(fp8_block, reference_block, initial_state, timed=False)
    step_results = []
    for _ in range(TIMED_SEQUENCES):
        step_results.extend(
            run_sequence(fp8_block, reference_block, initial_state, timed=True)
        )
    latencies = [result["latency_us"] for result in step_results]
    weight_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in fp8_block.parameters()
    )
    if weight_bytes >= MAX_WEIGHT_BYTES:
        raise RuntimeError(f"weight allocation exceeds limit: {weight_bytes}")
    return {
        **case,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "gate_up_gemm": {"m": case["m"], "k": HIDDEN_SIZE, "n": 2 * INTERMEDIATE_SIZE},
        "down_gemm": {"m": case["m"], "k": INTERMEDIATE_SIZE, "n": HIDDEN_SIZE},
        "steps": STEPS,
        "warmup_sequences": WARMUP_SEQUENCES,
        "timed_sequences": TIMED_SEQUENCES,
        "weight_bytes": weight_bytes,
        "median_latency_us": statistics.median(latencies),
        "p95_latency_us": percentile(latencies, 0.95),
        "min_latency_us": min(latencies),
        "max_latency_us": max(latencies),
        "raw_steps": step_results,
    }


def environment_report():
    device = torch.cuda.get_device_properties(0)
    return {
        "python": "/opt/venv/bin/python",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "gpu_name": device.name,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_bytes": device.total_memory,
        "image_id_sha256": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        "sglang_import_path": "/job/sglang/python/sglang/__init__.py",
        "native_torch_path": torch.__file__,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    torch.cuda.init()
    fp8_block, reference_block = build_blocks()
    results = {
        "label": "gfx942 native PyTorch FP8 reduced-block study",
        "kernel": "torch._scaled_mm",
        "input_dtype": "bfloat16 quantized to float8_e4m3fnuz",
        "weight_dtype": "float8_e4m3fnuz",
        "scale_dtype": "float32",
        "output_dtype": "bfloat16",
        "accuracy_gate": {
            "max_abs_error": ABS_ERROR_LIMIT,
            "max_relative_error": REL_ERROR_LIMIT,
            "finite": True,
        },
        "limits": {
            "workload_cases": len(CASES),
            "recurrent_steps": STEPS,
            "warmup_sequences_per_case": WARMUP_SEQUENCES,
            "timed_sequences_per_case": TIMED_SEQUENCES,
            "max_weight_bytes": MAX_WEIGHT_BYTES,
            "max_live_bytes": MAX_LIVE_BYTES,
            "wall_limit_seconds": 7200,
        },
        "timing_method": "CUDA events around each real block forward; median and p95 across all timed steps",
        "environment": environment_report(),
        "cases": [],
    }
    torch.cuda.reset_peak_memory_stats()
    for case in CASES:
        results["cases"].append(
            benchmark_case(case, fp8_block, reference_block)
        )
    peak_bytes = torch.cuda.max_memory_allocated()
    if peak_bytes >= MAX_LIVE_BYTES:
        raise RuntimeError(f"live allocation exceeds limit: {peak_bytes}")
    results["peak_allocated_bytes"] = peak_bytes
    results["elapsed_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        json.dump(results, output_file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
