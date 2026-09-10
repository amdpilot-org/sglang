import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton
from torch.profiler import ProfilerActivity, profile

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    moe_sum_reduce_torch_compile,
)


SHAPES = (1, 2, 4, 8, 16, 32)
TOPK = 2
HIDDEN_SIZE = 64
ROUTED_SCALING_FACTOR = 0.375
WARM_REPEATS = 3


def _timed_call(function, x, out):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    function(x, out, ROUTED_SCALING_FACTOR)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop)


def _run_shape(num_tokens):
    torch.manual_seed(10_000 + num_tokens)
    x = (
        torch.randn(num_tokens, TOPK, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    out = torch.full(
        (num_tokens, HIDDEN_SIZE),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    input_sentinel = x.clone()
    output_address = out.data_ptr()
    storage_address = out.untyped_storage().data_ptr()

    cold_ms = _timed_call(moe_sum_reduce_torch_compile, x, out)
    expected = (x.float().sum(dim=1) * ROUTED_SCALING_FACTOR).to(x.dtype)
    absolute_error = (out.float() - expected.float()).abs()
    checks = {
        "input_unchanged": torch.equal(x, input_sentinel),
        "output_sentinel_overwritten": not torch.isnan(out).any().item(),
        "output_address_stable_after_cold": out.data_ptr() == output_address,
        "output_storage_stable_after_cold": out.untyped_storage().data_ptr()
        == storage_address,
        "output_finite": torch.isfinite(out).all().item(),
    }

    warm_timings_ms = []
    for _ in range(WARM_REPEATS):
        out.fill_(float("nan"))
        warm_timings_ms.append(
            _timed_call(moe_sum_reduce_torch_compile, x, out)
        )
    checks.update(
        {
            "output_sentinel_overwritten_after_warm": not torch.isnan(out).any().item(),
            "output_address_stable_after_warm": out.data_ptr() == output_address,
            "output_storage_stable_after_warm": out.untyped_storage().data_ptr()
            == storage_address,
        }
    )
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    return {
        "num_tokens": num_tokens,
        "topk": TOPK,
        "hidden_size": HIDDEN_SIZE,
        "dtype": str(x.dtype).replace("torch.", ""),
        "checks": checks,
        "cold_ms": cold_ms,
        "warm_timings_ms": warm_timings_ms,
        "warm_median_ms": sorted(warm_timings_ms)[len(warm_timings_ms) // 2],
        "max_abs_error": absolute_error.max().item(),
        "mean_abs_error": absolute_error.mean().item(),
        "output_address": output_address,
    }


def _profile_dispatch():
    x = torch.randn(8, TOPK, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(8, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    moe_sum_reduce_torch_compile(x, out, ROUTED_SCALING_FACTOR)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        moe_sum_reduce_torch_compile(x, out, ROUTED_SCALING_FACTOR)
        torch.cuda.synchronize()
    dispatch = []
    for event in prof.key_averages():
        if (
            event.key.startswith("triton_poi_")
            or event.key.startswith("aten::")
            or event.key == "hipModuleLaunchKernel"
        ):
            dispatch.append(
                {
                    "name": event.key,
                    "device_time_us": event.self_device_time_total,
                    "cpu_time_us": event.cpu_time_total,
                }
            )
    return dispatch


def _unsupported_variants():
    results = {}
    x = torch.randn(2, TOPK, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    dtype_mismatch = torch.empty(2, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    try:
        moe_sum_reduce_torch_compile(x, dtype_mismatch, ROUTED_SCALING_FACTOR)
    except (RuntimeError, ValueError) as error:
        results["output_dtype_mismatch"] = {
            "supported": False,
            "error_type": type(error).__name__,
            "message": str(error).splitlines()[0],
        }
    else:
        results["output_dtype_mismatch"] = {"supported": True}

    shape_mismatch = torch.empty(2, HIDDEN_SIZE + 1, device="cuda", dtype=torch.bfloat16)
    try:
        moe_sum_reduce_torch_compile(x, shape_mismatch, ROUTED_SCALING_FACTOR)
    except (RuntimeError, ValueError) as error:
        results["output_shape_mismatch"] = {
            "supported": False,
            "error_type": type(error).__name__,
            "message": str(error).splitlines()[0],
        }
    else:
        results["output_shape_mismatch"] = {"supported": True}

    aliased_output = x.reshape(-1)[: 2 * HIDDEN_SIZE].view(2, HIDDEN_SIZE)
    try:
        moe_sum_reduce_torch_compile(x, aliased_output, ROUTED_SCALING_FACTOR)
    except (RuntimeError, ValueError) as error:
        results["overlapping_output_storage"] = {
            "supported": False,
            "error_type": type(error).__name__,
            "message": str(error).splitlines()[0],
        }
    else:
        results["overlapping_output_storage"] = {"supported": True}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    process_start = time.perf_counter()
    torch._dynamo.reset()
    first_start = time.perf_counter()
    shapes = [_run_shape(num_tokens) for num_tokens in SHAPES]
    first_result_elapsed = time.perf_counter() - first_start
    dispatch = _profile_dispatch()
    unsupported = _unsupported_variants()

    git_root = Path(__file__).resolve().parents[2]
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "label": "mirror-checkout compiled-path reuse experiment",
        "operation": "moe_sum_reduce_torch_compile",
        "source_commit": source_commit,
        "source_root": str(git_root),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
        },
        "image": {
            "name": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "local_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        },
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_module": torch.__file__,
        "triton": triton.__version__,
        "triton_module": triton.__file__,
        "sglang_module": __import__("sglang").__file__,
        "command": f"{sys.executable} {Path(__file__).resolve()} --output {args.output}",
        "reference": "independent fp32 x.sum(dim=1) * routed_scaling_factor, cast to bfloat16",
        "sentinel": "output pre-filled with NaN before cold and every warm call",
        "timing_method": (
            f"torch.cuda.Event elapsed_time; one cold call plus {WARM_REPEATS} "
            "warm calls per shape, no cache reset between shapes"
        ),
        "first_result_elapsed_seconds": first_result_elapsed,
        "process_start_to_first_result_seconds": time.perf_counter() - process_start,
        "shape_sequence": list(SHAPES),
        "results": shapes,
        "actual_dispatch": dispatch,
        "unsupported_variants": unsupported,
        "aliasing_policy": "supported calls use non-overlapping input and output storage; no aliasing variant is forced",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
