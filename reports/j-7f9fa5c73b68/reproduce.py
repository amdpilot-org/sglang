import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile


OUTPUT_PATH = Path(__file__).with_name("noncontiguous-index-select-results.json")
SENTINEL = -12345.0


def storage_ranges(tensor):
    storage = tensor.untyped_storage()
    start = storage.data_ptr()
    return start, start + storage.nbytes()


def ranges_overlap(left, right):
    left_start, left_end = storage_ranges(left)
    right_start, right_end = storage_ranges(right)
    return left_start < left_end and right_start < left_end


def check_contract(source, indices, output):
    if source.dim() != 2 or output.dim() != 2:
        raise ValueError("index_select experiment supports 2D source and output only")
    if indices.dtype != torch.int64:
        raise ValueError(f"index_select experiment requires int64 indices, got {indices.dtype}")
    if source.dtype != output.dtype:
        raise ValueError(
            f"index_select experiment requires matching source/output dtypes, got "
            f"{source.dtype} and {output.dtype}"
        )
    if source.device != indices.device or source.device != output.device:
        raise ValueError("index_select experiment requires source, indices, and output on one device")
    if tuple(output.shape) != (indices.numel(), source.shape[1]):
        raise ValueError(
            f"index_select output shape {tuple(output.shape)} does not match "
            f"({indices.numel()}, {source.shape[1]})"
        )
    if ranges_overlap(source, output):
        raise ValueError("index_select experiment forbids source/output aliasing")


def guarded_views(rows, cols, dtype, device):
    source_storage = torch.full(
        ((rows * 2) + 2, cols), SENTINEL, dtype=dtype, device=device
    )
    output_storage = torch.full(
        ((rows * 2) + 2, cols), SENTINEL, dtype=dtype, device=device
    )
    source = source_storage[1 : 1 + (rows * 2) : 2]
    output = output_storage[1 : 1 + (rows * 2) : 2]
    source.normal_()
    return source_storage, output_storage, source, output


def run_case(rows, cols, dtype, device):
    source_storage, output_storage, source, output = guarded_views(
        rows, cols, dtype, device
    )
    indices = torch.arange(rows, device=device, dtype=torch.int64)
    check_contract(source, indices, output)

    source_guard_before = source_storage[[0, -1]].clone()
    output_guard_before = output_storage[[0, -1]].clone()
    output_address_before = output.data_ptr()
    output_dtype_before = output.dtype

    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        torch.index_select(source, 0, indices, out=output)
        torch.cuda.synchronize()
    native_events = [
        {"name": event.key, "count": event.count}
        for event in profiler.key_averages()
        if str(event.device_type).endswith("CUDA") and event.self_device_time_total > 0
    ]

    reference = source.detach().cpu().index_select(0, indices.detach().cpu())
    exact = torch.equal(output.detach().cpu(), reference)
    unpacked_source = source.detach().contiguous()
    unpacked_output = torch.empty_like(unpacked_source)
    with profile(activities=[ProfilerActivity.CUDA]) as unpacked_profiler:
        torch.index_select(unpacked_source, 0, indices, out=unpacked_output)
        torch.cuda.synchronize()
    unpacked_native_events = [
        {"name": event.key, "count": event.count}
        for event in unpacked_profiler.key_averages()
        if str(event.device_type).endswith("CUDA") and event.self_device_time_total > 0
    ]
    unpacked_exact = torch.equal(output.detach().cpu(), unpacked_output.detach().cpu())
    max_abs_error = (output.detach().cpu() - reference).abs().max().item()
    source_guard_ok = torch.equal(source_storage[[0, -1]].cpu(), source_guard_before.cpu())
    output_guard_ok = torch.equal(output_storage[[0, -1]].cpu(), output_guard_before.cpu())

    timings = []
    for _ in range(3):
        torch.index_select(source, 0, indices, out=output)
    torch.cuda.synchronize()
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.index_select(source, 0, indices, out=output)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    return {
        "rows": rows,
        "cols": cols,
        "dtype": str(dtype),
        "source_stride": list(source.stride()),
        "output_stride": list(output.stride()),
        "source_contiguous": bool(source.is_contiguous()),
        "output_contiguous": bool(output.is_contiguous()),
        "exact_match": bool(exact),
        "unpacked_control_exact_match": bool(unpacked_exact),
        "unpacked_control_native_cuda_events": unpacked_native_events,
        "max_abs_error": max_abs_error,
        "source_sentinel_unchanged": bool(source_guard_ok),
        "output_sentinel_unchanged": bool(output_guard_ok),
        "output_address_unchanged": bool(output.data_ptr() == output_address_before),
        "output_dtype_unchanged": bool(output.dtype == output_dtype_before),
        "native_cuda_events": native_events,
        "timing_ms": timings,
        "timing_median_ms": sorted(timings)[len(timings) // 2],
    }


def expect_clear_failure(callable_object):
    try:
        callable_object()
    except (ValueError, RuntimeError) as error:
        return {"failed_clearly": True, "error_type": type(error).__name__, "message": str(error)}
    return {"failed_clearly": False, "error_type": None, "message": None}


def main():
    device = torch.device("cuda:0")
    start_wall = time.perf_counter()
    torch.cuda.init()
    cases = [
        run_case(rows, cols, torch.bfloat16, device)
        for rows, cols in ((64, 64), (256, 64), (64, 256), (256, 256))
    ]

    source_storage, output_storage, source, output = guarded_views(64, 64, torch.bfloat16, device)
    indices = torch.arange(64, device=device, dtype=torch.int64)
    dtype_failure = expect_clear_failure(lambda: check_contract(source, indices, output.to(torch.float32)))
    alias_failure = expect_clear_failure(lambda: check_contract(source, indices, source))

    try:
        gpu_identity = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showserial", "--showuniqueid"],
            text=True,
            timeout=10,
        )
    except Exception as error:
        gpu_identity = f"rocm-smi unavailable: {error}"

    properties = torch.cuda.get_device_properties(device)
    result = {
        "label": "persistent-checkout non-contiguous packed-versus-unpacked index_select",
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operation": "torch.index_select(source, 0, indices, out=output)",
        "representation": "source and output are non-contiguous row-strided views inside sentinel-protected packed storage",
        "reference": "independent CPU torch.index_select",
        "timing_method": "3 warmups then 10 CUDA-event timed calls per case",
        "contract": {
            "indices_dtype": "torch.int64",
            "source_output_dtype_must_match": True,
            "source_output_aliasing_forbidden": True,
            "static_output_address_preserved": True,
            "unsupported_variants_fail_before_dispatch": True,
        },
        "negative_controls": {
            "dtype_mismatch": dtype_failure,
            "source_output_aliasing": alias_failure,
        },
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "torch_module": torch.__file__,
            "torch_native_library": str(next((Path(torch.__file__).parent / "lib").glob("libtorch_hip*.so"))),
            "gpu_name": properties.name,
            "gpu_major_minor": [properties.major, properties.minor],
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_identity_raw": gpu_identity,
            "image_identity": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909 sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
        },
        "bounds": {
            "cases": len(cases),
            "calls_per_case": 13,
            "full_model_weights": False,
            "unbounded_or_burn_loops": False,
        },
        "first_gpu_execution_elapsed_s": time.perf_counter() - start_wall,
        "cases": cases,
        "all_cases_passed": all(
            case["exact_match"]
            and case["unpacked_control_exact_match"]
            and case["source_sentinel_unchanged"]
            and case["output_sentinel_unchanged"]
            and case["output_address_unchanged"]
            and case["output_dtype_unchanged"]
            for case in cases
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_cases_passed"]:
        raise SystemExit(1)
    if not all(case["native_cuda_events"] for case in cases):
        raise SystemExit(1)
    if not dtype_failure["failed_clearly"] or not alias_failure["failed_clearly"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
