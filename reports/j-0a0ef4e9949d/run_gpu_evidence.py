import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from sgl_kernel import merge_state_v2


OUTPUT_PATH = Path(__file__).with_name("gpu-evidence.json")
SHAPES = [(17, 3, 32), (64, 8, 64), (128, 8, 128)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TIMING_WARMUPS = 3
TIMING_REPEATS = 20


def reference(prefix_output, prefix_lse, suffix_output, suffix_lse):
    prefix_lse = prefix_lse.to(torch.float64).clone()
    suffix_lse = suffix_lse.to(torch.float64).clone()
    prefix_lse = torch.where(
        torch.isinf(prefix_lse), -torch.inf * torch.ones_like(prefix_lse), prefix_lse
    )
    suffix_lse = torch.where(
        torch.isinf(suffix_lse), -torch.inf * torch.ones_like(suffix_lse), suffix_lse
    )
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    prefix_lse = prefix_lse - max_lse
    suffix_lse = suffix_lse - max_lse
    prefix_sum_exp = torch.exp(prefix_lse)
    suffix_sum_exp = torch.exp(suffix_lse)
    sum_exp = prefix_sum_exp + suffix_sum_exp
    output_lse = torch.log(sum_exp) + max_lse
    prefix_scale = (prefix_sum_exp / sum_exp).unsqueeze(2)
    suffix_scale = (suffix_sum_exp / sum_exp).unsqueeze(2)
    output = (
        prefix_output.to(torch.float64) * prefix_scale
        + suffix_output.to(torch.float64) * suffix_scale
    )
    return output, output_lse


def sentinel_output(shape, dtype, sentinel, guard_count=11):
    numel = torch.Size(shape).numel()
    storage = torch.full((numel + guard_count,), sentinel, dtype=dtype, device="cuda")
    return storage[:numel].view(shape), storage[numel:]


def make_case(shape, dtype, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prefix_base = torch.randn(shape, generator=generator, dtype=torch.float32)
    suffix_base = torch.randn(shape, generator=generator, dtype=torch.float32)
    prefix_lse_base = torch.randn(
        shape[:2], generator=generator, dtype=torch.float32
    )
    suffix_lse_base = torch.randn(
        shape[:2], generator=generator, dtype=torch.float32
    )
    return (
        prefix_base.to(device="cuda", dtype=dtype),
        prefix_lse_base.to(device="cuda", dtype=torch.float32),
        suffix_base.to(device="cuda", dtype=dtype),
        suffix_lse_base.to(device="cuda", dtype=torch.float32),
    )


def timing(prefix_output, prefix_lse, suffix_output, suffix_lse, dtype):
    v_merged = torch.empty_like(prefix_output)
    s_merged = torch.empty_like(prefix_lse)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(TIMING_WARMUPS):
        merge_state_v2(
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            v_merged,
            s_merged,
        )
    torch.cuda.synchronize()
    total_ms = 0.0
    for _ in range(TIMING_REPEATS):
        start.record()
        merge_state_v2(
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            v_merged,
            s_merged,
        )
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / TIMING_REPEATS


def error_message(callable):
    try:
        callable()
    except Exception as exc:
        return {"type": type(exc).__name__, "message": str(exc)}
    return None


def main():
    started = time.perf_counter()
    first_execution_started = None
    first_gpu_execution_elapsed_seconds = None
    cases = []
    dispatch = []
    for dtype_index, dtype in enumerate(DTYPES):
        for shape_index, shape in enumerate(SHAPES):
            prefix_output, prefix_lse, suffix_output, suffix_lse = make_case(
                shape, dtype, 1715 + dtype_index * 100 + shape_index
            )
            expected, expected_lse = reference(
                prefix_output.cpu(),
                prefix_lse.cpu(),
                suffix_output.cpu(),
                suffix_lse.cpu(),
            )
            expected_cast = expected.to(dtype)
            expected_lse_cast = expected_lse.to(torch.float32)
            v_merged, v_guard = sentinel_output(shape, dtype, float("nan"))
            s_merged, s_guard = sentinel_output(
                shape[:2], torch.float32, -12345.0
            )
            if first_execution_started is None:
                first_execution_started = time.perf_counter()
            returned_v, returned_lse = merge_state_v2(
                prefix_output,
                prefix_lse,
                suffix_output,
                suffix_lse,
                v_merged,
                s_merged,
            )
            torch.cuda.synchronize()
            if dtype_index == 0 and shape_index == 0:
                first_gpu_execution_elapsed_seconds = (
                    time.perf_counter() - first_execution_started
                )
            actual_cpu = returned_v.cpu().to(torch.float64)
            actual_lse_cpu = returned_lse.cpu().to(torch.float64)
            tolerance = (
                (2e-6, 2e-6)
                if dtype == torch.float32
                else (1e-3, 1e-3)
                if dtype == torch.float16
                else (1e-2, 1e-2)
            )
            torch.testing.assert_close(
                returned_v.cpu(), expected_cast, atol=tolerance[0], rtol=tolerance[1]
            )
            torch.testing.assert_close(
                returned_lse.cpu(),
                expected_lse_cast,
                atol=2e-6,
                rtol=2e-6,
            )
            cases.append(
                {
                    "dtype": str(dtype).removeprefix("torch."),
                    "shape": {"tokens": shape[0], "heads": shape[1], "head_size": shape[2]},
                    "max_abs_diff_vs_float64": float(
                        (actual_cpu - expected).abs().max().item()
                    ),
                    "max_abs_diff_vs_dtype_cast_reference": float(
                        (actual_cpu - expected_cast.to(torch.float64).cpu()).abs().max().item()
                    ),
                    "max_lse_abs_diff_vs_float64": float(
                        (actual_lse_cpu - expected_lse).abs().max().item()
                    ),
                    "v_guard_unchanged": bool(torch.isnan(v_guard).all().item()),
                    "s_guard_unchanged": bool((s_guard == -12345.0).all().item()),
                    "average_ms": timing(
                        prefix_output,
                        prefix_lse,
                        suffix_output,
                        suffix_lse,
                        dtype,
                    ),
                }
            )

        prefix_output, prefix_lse, suffix_output, suffix_lse = make_case(
            SHAPES[0], dtype, 228
        )
        v_merged = torch.empty_like(prefix_output)
        s_merged = torch.empty_like(prefix_lse)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            merge_state_v2(
                prefix_output,
                prefix_lse,
                suffix_output,
                suffix_lse,
                v_merged,
                s_merged,
            )
            torch.cuda.synchronize()
        dispatch.append(
            {
                "dtype": str(dtype).removeprefix("torch."),
                "events": [
                    event.key
                    for event in profiler.key_averages()
                    if "merge_attn" in event.key.lower()
                    or event.device_type == torch.autograd.DeviceType.CUDA
                ],
            }
        )

    prefix_output, prefix_lse, suffix_output, suffix_lse = make_case(
        SHAPES[0], torch.float32, 5419
    )
    expected, expected_lse = reference(
        prefix_output.cpu(),
        prefix_lse.cpu(),
        suffix_output.cpu(),
        suffix_lse.cpu(),
    )
    returned_v, returned_lse = merge_state_v2(
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_output,
        prefix_lse,
    )
    torch.testing.assert_close(
        returned_v.cpu(), expected.to(torch.float32), atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        returned_lse.cpu(), expected_lse.to(torch.float32), atol=2e-6, rtol=2e-6
    )
    aliasing = {
        "v_output_aliases_prefix": returned_v.data_ptr() == prefix_output.data_ptr(),
        "s_output_aliases_prefix_lse": returned_lse.data_ptr() == prefix_lse.data_ptr(),
    }

    v_merged = torch.empty(SHAPES[0], dtype=torch.float32, device="cuda")
    s_merged = torch.empty(SHAPES[0][:2], dtype=torch.float32, device="cuda")
    v_address = v_merged.data_ptr()
    s_address = s_merged.data_ptr()
    for seed in (1715, 228):
        prefix_output, prefix_lse, suffix_output, suffix_lse = make_case(
            SHAPES[0], torch.float32, seed
        )
        merge_state_v2(
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            v_merged,
            s_merged,
        )
    static_storage = {
        "v_address_unchanged": v_merged.data_ptr() == v_address,
        "s_address_unchanged": s_merged.data_ptr() == s_address,
    }

    valid = make_case(SHAPES[0], torch.float32, 1715)
    unsupported = {
        "mixed_value_dtype": error_message(
            lambda: merge_state_v2(
                valid[0],
                valid[1],
                valid[2],
                valid[3],
                torch.empty(SHAPES[0], dtype=torch.float16, device="cuda"),
                torch.empty(SHAPES[0][:2], dtype=torch.float32, device="cuda"),
            )
        ),
        "float64_dispatch": error_message(
            lambda: merge_state_v2(
                valid[0].to(torch.float64),
                valid[1],
                valid[2].to(torch.float64),
                valid[3],
                torch.empty(SHAPES[0], dtype=torch.float64, device="cuda"),
                torch.empty(SHAPES[0][:2], dtype=torch.float32, device="cuda"),
            )
        ),
        "wrong_output_shape": error_message(
            lambda: merge_state_v2(
                valid[0],
                valid[1],
                valid[2],
                valid[3],
                torch.empty(
                    (SHAPES[0][0] + 1, *SHAPES[0][1:]),
                    dtype=torch.float32,
                    device="cuda",
                ),
                torch.empty(SHAPES[0][:2], dtype=torch.float32, device="cuda"),
            )
        ),
    }

    try:
        import flashinfer
        flashinfer_path = flashinfer.__file__
    except Exception as exc:
        flashinfer_path = f"{type(exc).__name__}: {exc}"

    result = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "count": torch.cuda.device_count(),
        },
        "torch": {
            "version": torch.__version__,
            "file": torch.__file__,
            "hip": torch.version.hip,
        },
        "native": {
            "sgl_kernel_file": __import__("sgl_kernel").__file__,
            "common_ops_file": __import__("sgl_kernel").common_ops.__file__,
            "op_schema": str(torch.ops.sgl_kernel.merge_state_v2.default._schema),
        },
        "flashinfer_path": flashinfer_path,
        "image": {
            "required_local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
            "container_hostname": os.uname().nodename,
        },
        "reference": "independent float64 PyTorch merge from exact dtype-represented inputs",
        "sentinel": "NaN/-12345 guard elements after each output view",
        "timing_method": f"torch.cuda.Event; {TIMING_WARMUPS} warmups and {TIMING_REPEATS} timed calls per case",
        "timing_matrix_size": len(cases),
        "first_gpu_execution_elapsed_seconds": first_gpu_execution_elapsed_seconds,
        "cases": cases,
        "native_dispatch": dispatch,
        "aliasing": aliasing,
        "static_storage_reuse": static_storage,
        "unsupported_contracts": unsupported,
        "commands": [
            "PYTHONPATH=/tmp/sglang-cache-j-0a0ef4e9949d/lib /opt/venv/bin/python reports/j-0a0ef4e9949d/run_gpu_evidence.py",
            "PYTHONPATH=/tmp/sglang-cache-j-0a0ef4e9949d/lib /opt/venv/bin/python -m pytest python/sglang/kernels/aot/tests/test_merge_state_v2.py -k 'float32_contracts or aliasing_and_static_storage_reuse or rejects_unsupported_contracts'",
        ],
        "baseline": "/job/baseline-first.json",
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
