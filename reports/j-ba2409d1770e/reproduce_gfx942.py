import inspect
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

CACHE_ROOT = Path("/tmp/sglang-cache-j-ba2409d1770e")
TRITON_CACHE = CACHE_ROOT / "checkout-triton"
TRITON_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(TRITON_CACHE)

from sglang.srt.layers.layernorm import Gemma3RMSNorm  # noqa: E402


OUTPUT = Path(__file__).with_name("gfx942-results.json")
DEVICE = torch.device("cuda:0")
EPS = 1e-6
SHAPES = ((1, 1152), (64, 1152), (257, 1152))
WARMUPS = 3
MEASURED_CALLS = 10


def independent_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    cpu_x = x.detach().cpu().double()
    cpu_weight = weight.detach().cpu().double()
    scale = torch.rsqrt(cpu_x.pow(2).mean(-1, keepdim=True) + EPS)
    return (cpu_x * scale * (1.0 + cpu_weight)).to(x.dtype).to(DEVICE)


def comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.detach()
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-6)
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "max_rel_diff": float((difference / denominator).max()),
        "allclose_1e_2": bool(
            torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)
        ),
    }


def time_calls(function) -> dict:
    for _ in range(WARMUPS):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(MEASURED_CALLS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "warmups": WARMUPS,
        "measured_calls": MEASURED_CALLS,
        "samples_milliseconds": samples,
        "mean_milliseconds": statistics.fmean(samples),
        "min_milliseconds": min(samples),
        "max_milliseconds": max(samples),
    }


def main() -> None:
    started = time.perf_counter()
    torch.cuda.set_device(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(32807)
    results = []
    output_addresses = []

    for batch_index, (rows, columns) in enumerate(SHAPES):
        x = (
            torch.randn(
                (rows, columns),
                generator=generator,
                device=DEVICE,
                dtype=torch.float32,
            )
            * (0.02 + 0.01 * batch_index)
        ).to(torch.bfloat16)
        weight = (
            torch.randn(
                columns,
                generator=generator,
                device=DEVICE,
                dtype=torch.float32,
            )
            * 0.1
        )
        sentinel = torch.full(
            (rows, columns), -12345.0, device=DEVICE, dtype=torch.bfloat16
        )
        input_copy = x.detach().clone()
        input_address = x.data_ptr()
        sentinel_address = sentinel.data_ptr()

        module = Gemma3RMSNorm(columns)
        module.weight.data = weight.to(torch.bfloat16)
        module_weight = module.weight.data
        actual = module.forward_hip(x)
        torch.cuda.synchronize()
        expected = independent_reference(x, module_weight)
        output_addresses.append(actual.data_ptr())
        result = comparison(actual, expected)
        result.update(
            {
                "shape": list(x.shape),
                "input_dtype": str(x.dtype),
                "weight_dtype": str(module_weight.dtype),
                "output_dtype": str(actual.dtype),
                "output_device": str(actual.device),
                "aliases_input": actual.data_ptr() == input_address,
                "aliases_sentinel_candidate": actual.data_ptr()
                == sentinel_address,
                "sentinel_candidate_unchanged": bool(
                    torch.equal(
                        sentinel,
                        torch.full_like(sentinel, -12345.0),
                    )
                ),
                "input_unchanged": bool(torch.equal(x.detach(), input_copy)),
            }
        )
        result["timing"] = time_calls(
                lambda current_x=x, current_module=module: current_module.forward_hip(
                    current_x
                )
        )
        results.append(result)

    reuse = torch.empty((64, 1152), device=DEVICE, dtype=torch.bfloat16)
    reuse.fill_(-54321.0)
    reuse_input = torch.randn(
        (64, 1152), generator=generator, device=DEVICE, dtype=torch.float32
    ).to(torch.bfloat16)
    reuse_weight = torch.randn(
        1152, generator=generator, device=DEVICE, dtype=torch.float32
    )
    try:
        reuse_module = Gemma3RMSNorm(1152)
        reuse_module.weight.data = reuse_weight.to(torch.bfloat16)
        reuse_module.forward_hip(reuse_input, out=reuse)
        torch.cuda.synchronize()
        reuse_attempt = {
            "raised": False,
            "note": "unexpectedly accepted an out argument",
        }
    except Exception as error:
        reuse_attempt = {
            "raised": True,
            "type": type(error).__name__,
            "message": str(error),
        }

    probe_x = torch.randn(
        (64, 1152), generator=generator, device=DEVICE, dtype=torch.float32
    ).to(torch.bfloat16)
    probe_weight = torch.randn(
        1152, generator=generator, device=DEVICE, dtype=torch.float32
    ).to(torch.bfloat16)
    probe_module = Gemma3RMSNorm(1152)
    probe_module.weight.data = probe_weight
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        probe_module.forward_hip(probe_x)
        torch.cuda.synchronize()
    native_kernels = [
        event.name
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and "Synchronize" not in event.name
    ]

    report = {
        "label": "persistent-checkout gfx942 probe",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": {
            "name": torch.cuda.get_device_name(DEVICE),
            "capability": torch.cuda.get_device_capability(DEVICE),
            "device": str(DEVICE),
        },
        "torch": {"version": torch.__version__, "file": torch.__file__},
        "operation": {
            "class": inspect.getsourcefile(Gemma3RMSNorm),
            "method": "Gemma3RMSNorm.forward_hip",
            "signature": str(inspect.signature(Gemma3RMSNorm.forward_hip)),
            "actual_dispatch": "Gemma3RMSNorm.forward_hip -> forward_native",
        },
        "reference": "independent CPU float64 Gemma RMSNorm",
        "fresh_output_cases": results,
        "fresh_output_addresses_distinct_across_shapes": len(
            set(output_addresses)
        )
        == len(output_addresses),
        "documented_reuse_attempt": reuse_attempt,
        "native_kernels": native_kernels,
        "timing_method": (
            "3 warmups then 10 measured calls per shape; CUDA events "
            "around one real Gemma3RMSNorm.forward_hip invocation; milliseconds"
        ),
        "bounds": {
            "shapes": [list(shape) for shape in SHAPES],
            "warmups_per_shape": WARMUPS,
            "measured_calls_per_shape": MEASURED_CALLS,
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
