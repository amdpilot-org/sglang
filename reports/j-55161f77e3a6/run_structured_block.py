from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.norm.group_norm_silu_twopass_triton import (
    group_norm_silu_4d,
)
from sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler import ResBlock


CASES = (
    "zeros",
    "tiny",
    "mixed_magnitudes",
    "cancellation",
    "skewed_sparse",
)
SHAPE = (2, 64, 32, 32)
DTYPE = torch.bfloat16
GROUPS = 32
EPS = 1e-5
ITERS = 30


def structured_input(case: str) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(55161)
    if case == "zeros":
        return torch.zeros(SHAPE, device="cuda", dtype=DTYPE)
    if case == "tiny":
        values = torch.rand(SHAPE, device="cuda", generator=generator)
        return values.mul_(2e-8).sub_(1e-8).to(DTYPE)
    if case == "mixed_magnitudes":
        count = math.prod(SHAPE)
        values = torch.logspace(-4, 4, count, device="cuda", dtype=torch.float32)
        signs = torch.where(
            torch.rand(SHAPE, device="cuda", generator=generator) < 0.5,
            -1.0,
            1.0,
        )
        return values.reshape(SHAPE).mul_(signs).to(DTYPE)
    if case == "cancellation":
        half = torch.rand(
            (SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3] // 2),
            device="cuda",
            generator=generator,
        ).sub_(0.5)
        return torch.cat((half, -half), dim=3).to(DTYPE)
    if case == "skewed_sparse":
        values = torch.rand(SHAPE, device="cuda", generator=generator)
        mask = values.lt(0.001)
        return values.mul_(mask.float().mul_(1000.0)).to(DTYPE)
    raise AssertionError(case)


def manual_group_norm_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    apply_silu: bool,
) -> torch.Tensor:
    original_dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(x_float.shape[0], num_groups, -1)
    mean = grouped.mean(dim=-1, keepdim=True)
    variance = grouped.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (grouped - mean) * torch.rsqrt(variance + eps)
    normalized = normalized.reshape_as(x_float)
    normalized = normalized * weight.float().reshape(1, -1, 1, 1)
    normalized = normalized + bias.float().reshape(1, -1, 1, 1)
    if apply_silu:
        normalized = F.silu(normalized)
    return normalized.to(original_dtype)


def independent_reference(block: ResBlock, x: torch.Tensor) -> torch.Tensor:
    residual = x
    conv1 = F.conv2d(
        x,
        block.conv1.weight,
        bias=block.conv1.bias,
        stride=block.conv1.stride,
        padding=block.conv1.padding,
    )
    normalized1 = manual_group_norm_silu(
        conv1,
        block.norm1.weight,
        block.norm1.bias,
        block.norm1.num_groups,
        block.norm1.eps,
        apply_silu=True,
    )
    conv2 = F.conv2d(
        normalized1,
        block.conv2.weight,
        bias=block.conv2.bias,
        stride=block.conv2.stride,
        padding=block.conv2.padding,
    )
    normalized2 = manual_group_norm_silu(
        conv2,
        block.norm2.weight,
        block.norm2.bias,
        block.norm2.num_groups,
        block.norm2.eps,
        apply_silu=False,
    )
    return F.silu(normalized2 + residual).to(x.dtype)


def timed(fn, count: int = ITERS) -> list[float]:
    samples = []
    for _ in range(count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(55161)
    block = ResBlock(channels=64, dims=2).to(
        device=torch.device("cuda"), dtype=DTYPE
    )
    block.eval()
    results = []
    for case in CASES:
        x = structured_input(case)
        with torch.no_grad():
            actual = block(x)
            reference = independent_reference(block, x)
            diff = (actual.float() - reference.float()).abs()
            block_ms = timed(lambda: block(x))
            reference_ms = timed(lambda: independent_reference(block, x))
            contiguous_ms = timed(lambda: x.contiguous())
            channels_last_ms = timed(
                lambda: x.to(memory_format=torch.channels_last).contiguous()
            )
            channels_last = x.to(memory_format=torch.channels_last)
            channels_last_to_nchw_ms = timed(lambda: channels_last.contiguous())
            twopass_ms = timed(
                lambda: group_norm_silu_4d(
                    channels_last,
                    block.norm1.weight,
                    block.norm1.bias,
                    GROUPS,
                    EPS,
                    apply_silu=True,
                )
            )
        results.append(
            {
                "case": case,
                "finite": bool(torch.isfinite(actual).all().item()),
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
                "block_ms_median": median(block_ms),
                "reference_ms_median": median(reference_ms),
                "contiguous_conversion_ms_median": median(contiguous_ms),
                "nchw_to_channels_last_ms_median": median(channels_last_ms),
                "channels_last_to_nchw_ms_median": median(channels_last_to_nchw_ms),
                "channels_last_twopass_ms_median": median(twopass_ms),
            }
        )
        print(case, results[-1], flush=True)
    payload = {
        "label": "mirror-checkout structured-input result",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 3),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "count": torch.cuda.device_count(),
        },
        "python": os.sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "sglang_import_path": os.path.dirname(__import__("sglang").__file__),
        "torch_path": os.path.dirname(torch.__file__),
        "torch_lib_path": os.path.join(os.path.dirname(torch.__file__), "lib"),
        "shape": list(SHAPE),
        "dtype": str(DTYPE),
        "groups": GROUPS,
        "eps": EPS,
        "iterations": ITERS,
        "timing_method": "CUDA events, 30 calls per case, median reported",
        "reference_method": "fp32 manual GroupNorm statistics and SiLU, bf16 output cast",
        "numerical_gate": "all finite and torch.testing.assert_close(atol=0.07, rtol=0.02)",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
