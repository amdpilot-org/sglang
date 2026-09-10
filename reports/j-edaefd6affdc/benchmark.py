from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import can_use_wan_rmsnorm_silu, wan_rmsnorm_silu
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    WanCausalConv3d,
    WanRMS_norm,
)


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    channels: int
    frames: int
    height: int
    width: int


CASES = [
    Case("c96_128", 1, 96, 4, 128, 128),
    Case("c96_256", 1, 96, 4, 256, 256),
    Case("c96_384", 1, 96, 4, 384, 384),
    Case("c96_512", 1, 96, 4, 512, 512),
    Case("c96_512_b2", 2, 96, 4, 512, 512),
    Case("c384_256", 1, 384, 4, 256, 256),
]


def cuda_time(fn, warmups: int = 3, iterations: int = 10) -> tuple[float, float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    total = start.elapsed_time(end) / 1000.0
    return total / iterations, total


def wall_time(fn) -> float:
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def make_block(channels: int, device: str, dtype: torch.dtype):
    conv1 = WanCausalConv3d(channels, channels, 3, padding=1).to(device=device, dtype=dtype)
    conv2 = WanCausalConv3d(channels, channels, 3, padding=1).to(device=device, dtype=dtype)
    norm = WanRMS_norm(channels, images=False).to(device=device, dtype=dtype)
    with torch.no_grad():
        for conv in (conv1, conv2):
            conv.weight.normal_(0.0, 0.025)
            conv.bias.zero_()
        norm.gamma.normal_(1.0, 0.025)
    return conv1, conv2, norm


def independent_reference(x, conv1, conv2, norm):
    scale = float(norm.scale)
    gamma = norm.gamma
    h = F.pad(x, (1, 1, 1, 1, 2, 0))
    h = F.conv3d(h, conv1.weight, conv1.bias)
    h = F.silu(F.normalize(h, dim=1) * scale * gamma)
    h = F.pad(h, (1, 1, 1, 1, 2, 0))
    return F.conv3d(h, conv2.weight, conv2.bias)


def fused_path(x, conv1, conv2, norm):
    h = conv1(x)
    h = h.contiguous(memory_format=torch.channels_last_3d)
    if not can_use_wan_rmsnorm_silu(h, norm.gamma, None):
        raise RuntimeError("fused RMSNorm+SiLU path rejected the convolution output")
    h = wan_rmsnorm_silu(h, norm.gamma, rms_scale=float(norm.scale))
    return conv2(h)


def error_metrics(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "rmse": diff.pow(2).mean().sqrt().item(),
    }


def tensor_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def run_case(case: Case, args):
    torch.manual_seed(23494)
    device = "cuda"
    dtype = torch.bfloat16
    shape = (case.batch, case.channels, case.frames, case.height, case.width)

    contiguous_x = torch.randn(shape, device=device, dtype=dtype)
    channels_last_x = contiguous_x.contiguous(memory_format=torch.channels_last_3d)
    conv1, conv2, norm = make_block(case.channels, device, dtype)

    contiguous_conv1 = WanCausalConv3d(case.channels, case.channels, 3, padding=1).to(device=device, dtype=dtype)
    contiguous_conv2 = WanCausalConv3d(case.channels, case.channels, 3, padding=1).to(device=device, dtype=dtype)
    contiguous_norm = WanRMS_norm(case.channels, images=False).to(device=device, dtype=dtype)
    contiguous_conv1.load_state_dict(conv1.state_dict())
    contiguous_conv2.load_state_dict(conv2.state_dict())
    contiguous_norm.load_state_dict(norm.state_dict())

    channels_last_conv1 = WanCausalConv3d(case.channels, case.channels, 3, padding=1).to(device=device, dtype=dtype)
    channels_last_conv2 = WanCausalConv3d(case.channels, case.channels, 3, padding=1).to(device=device, dtype=dtype)
    channels_last_norm = WanRMS_norm(case.channels, images=False).to(device=device, dtype=dtype)
    channels_last_conv1.load_state_dict(conv1.state_dict())
    channels_last_conv2.load_state_dict(conv2.state_dict())
    channels_last_norm.load_state_dict(norm.state_dict())
    channels_last_conv1 = channels_last_conv1.to(memory_format=torch.channels_last_3d)
    channels_last_conv2 = channels_last_conv2.to(memory_format=torch.channels_last_3d)

    with torch.no_grad():
        contiguous_fn = lambda: independent_reference(
            contiguous_x, contiguous_conv1, contiguous_conv2, contiguous_norm
        )
        channels_last_fn = lambda: independent_reference(
            channels_last_x, channels_last_conv1, channels_last_conv2, channels_last_norm
        )
        fused_fn = lambda: fused_path(
            channels_last_x, channels_last_conv1, channels_last_conv2, channels_last_norm
        )

        cold_contiguous = wall_time(contiguous_fn)
        cold_channels_last = wall_time(channels_last_fn)
        cold_fused = wall_time(fused_fn)

        contiguous_mean, contiguous_total = cuda_time(contiguous_fn, args.warmups, args.iterations)
        channels_last_mean, channels_last_total = cuda_time(channels_last_fn, args.warmups, args.iterations)
        fused_mean, fused_total = cuda_time(fused_fn, args.warmups, args.iterations)

        expected = independent_reference(
            channels_last_x, channels_last_conv1, channels_last_conv2, channels_last_norm
        )
        actual = fused_path(
            channels_last_x, channels_last_conv1, channels_last_conv2, channels_last_norm
        )
        metrics = error_metrics(actual, expected)

        to_channels_last_mean, _ = cuda_time(
            lambda: contiguous_x.contiguous(memory_format=torch.channels_last_3d),
            args.warmups,
            args.iterations,
        )
        to_contiguous_mean, _ = cuda_time(
            lambda: channels_last_x.contiguous(),
            args.warmups,
            args.iterations,
        )
        weight_to_channels_last_mean, _ = cuda_time(
            lambda: contiguous_conv1.to(memory_format=torch.channels_last_3d),
            args.warmups,
            args.iterations,
        )
        conv_output = channels_last_conv1(channels_last_x)
        post_conv_to_channels_last_mean, _ = cuda_time(
            lambda: conv_output.contiguous(memory_format=torch.channels_last_3d),
            args.warmups,
            args.iterations,
        )

    tokens = case.batch * case.frames * case.height * case.width
    output_elements = case.batch * case.channels * case.frames * case.height * case.width
    weight_storage = tensor_bytes(
        contiguous_conv1.weight,
        contiguous_conv1.bias,
        contiguous_conv2.weight,
        contiguous_conv2.bias,
        contiguous_norm.gamma,
    )
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    return {
        "name": case.name,
        "shape": list(shape),
        "tokens": tokens,
        "dtype": str(dtype),
        "paths": {
            "contiguous_reference": {
                "cold_seconds": cold_contiguous,
                "warm_mean_seconds": contiguous_mean,
                "warm_total_seconds": contiguous_total,
                "tokens_per_second": tokens / contiguous_mean,
                "output_elements_per_second": output_elements / contiguous_mean,
            },
            "channels_last_reference": {
                "cold_seconds": cold_channels_last,
                "warm_mean_seconds": channels_last_mean,
                "warm_total_seconds": channels_last_total,
                "tokens_per_second": tokens / channels_last_mean,
                "output_elements_per_second": output_elements / channels_last_mean,
            },
            "channels_last_fused": {
                "cold_seconds": cold_fused,
                "warm_mean_seconds": fused_mean,
                "warm_total_seconds": fused_total,
                "tokens_per_second": tokens / fused_mean,
                "output_elements_per_second": output_elements / fused_mean,
            },
        },
        "layout_conversion": {
            "activation_to_channels_last_seconds": to_channels_last_mean,
            "activation_to_contiguous_seconds": to_contiguous_mean,
            "weight_to_channels_last_seconds": weight_to_channels_last_mean,
            "post_conv_to_channels_last_seconds": post_conv_to_channels_last_mean,
        },
        "accuracy": metrics,
        "accuracy_gate": "max_abs_error <= 0.15 and rmse <= 0.03",
        "accuracy_gate_passed": metrics["max_abs_error"] <= 0.15 and metrics["rmse"] <= 0.03,
        "memory": {
            "weight_storage_bytes": weight_storage,
            "weight_storage_gib": weight_storage / 1024**3,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU is required")
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

    results = {
        "schema_version": 1,
        "study": "gfx942 synthetic diffusion feature block",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "python_executable": os.path.abspath(sys.executable),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
        "operator_imports": [
            "sglang.kernels.ops.diffusion.wan_rmsnorm_silu",
            "sglang.multimodal_gen.runtime.models.vaes.wanvae.WanCausalConv3d",
        ],
        "reference": "independent F.pad + F.conv3d + F.normalize + F.silu formula",
        "timing_method": f"{args.warmups} warmups followed by {args.iterations} timed forwards using CUDA events",
        "limits": {
            "maximum_cases": 6,
            "maximum_weight_storage_bytes": 4 * 1024**3,
            "maximum_live_allocations_bytes": 48 * 1024**3,
            "wall_clock_seconds": 7200,
        },
        "cases": [],
    }

    for case in CASES:
        torch.cuda.reset_peak_memory_stats()
        result = run_case(case, args)
        results["cases"].append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    import sys

    main()
