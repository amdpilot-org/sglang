from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from sglang.kernels.ops.elementwise.elementwise import (
    _fused_sigmoid_mul_kernel,
    fused_sigmoid_mul,
)

DTYPE = torch.float16
SENTINEL = -12345.0
SHAPES = ((1, 2048), (3, 2048), (7, 2048))
WARM_MEASUREMENTS = 5
GRAPH_REPLAYS = 4


def module_path(name: str) -> str | None:
    specification = importlib.util.find_spec(name)
    return specification.origin if specification else None


def cache_size() -> int:
    device_cache = _fused_sigmoid_mul_kernel.device_caches.get(0)
    if not device_cache:
        return 0
    return len(device_cache[0])


def reference(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (
        attn_output.detach()
        .cpu()
        .double()
        .mul(torch.sigmoid(gate.detach().cpu().double()))
        .to(DTYPE)
    )


def timed_call(attn_output: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    started_ns = time.perf_counter_ns()
    output = fused_sigmoid_mul(attn_output, gate, inplace=False)
    torch.cuda.synchronize()
    return output, (time.perf_counter_ns() - started_ns) / 1_000_000


def event_time_ms(function) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    function()
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event)


def profiler_events(function) -> list[dict[str, object]]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        function()
        torch.cuda.synchronize()
    events = []
    for event in profile.key_averages():
        if event.self_device_time_total > 0 or "kernel" in event.key.lower() or "hip" in event.key.lower():
            events.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "self_device_time_us": event.self_device_time_total,
                    "device_time_us": event.device_time_total,
                }
            )
    return events


def make_case(shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260910)
    attn_output = torch.randn(
        shape, generator=generator, dtype=torch.float32
    ).to(device="cuda", dtype=DTYPE)
    gate = torch.randn(
        shape, generator=generator, dtype=torch.float32
    ).to(device="cuda", dtype=DTYPE)
    return attn_output, gate, reference(attn_output, gate)


def run_shape(shape: tuple[int, int]) -> dict[str, object]:
    attn_output, gate, expected = make_case(shape)
    output, first_call_ms = timed_call(attn_output, gate)
    warm_ms = []
    for _ in range(WARM_MEASUREMENTS):
        warm_output, elapsed_ms = timed_call(attn_output, gate)
        warm_ms.append(elapsed_ms)
        del warm_output

    absolute_error = (output.cpu().double() - expected.double()).abs().max().item()
    torch.testing.assert_close(
        output.cpu(), expected, rtol=1.0e-2, atol=1.0e-2
    )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fused_sigmoid_mul(attn_output, gate, inplace=False)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fused_sigmoid_mul(attn_output, gate, inplace=False)
    addresses = (
        attn_output.data_ptr(),
        gate.data_ptr(),
        graph_output.data_ptr(),
    )
    guard = torch.full(shape, SENTINEL, device="cuda", dtype=DTYPE)
    graph_output.fill_(SENTINEL)
    torch.cuda.synchronize()
    replay_event_ms = [
        event_time_ms(graph.replay) for _ in range(GRAPH_REPLAYS)
    ]
    torch.cuda.synchronize()

    torch.testing.assert_close(
        graph_output.cpu(), expected, rtol=1.0e-2, atol=1.0e-2
    )
    assert guard.cpu().eq(SENTINEL).all()
    assert addresses == (
        attn_output.data_ptr(),
        gate.data_ptr(),
        graph_output.data_ptr(),
    )

    return {
        "shape": list(shape),
        "dtype": str(DTYPE),
        "first_call_ms": first_call_ms,
        "warm_wall_ms": warm_ms,
        "warm_wall_median_ms": sorted(warm_ms)[len(warm_ms) // 2],
        "reference_max_abs_error": absolute_error,
        "graph_replay_event_ms": replay_event_ms,
        "graph_replays": GRAPH_REPLAYS,
        "static_addresses": [hex(address) for address in addresses],
        "guard_unchanged": True,
        "output_sentinel_overwritten": True,
    }


def unsupported_torch_compile_result() -> dict[str, object]:
    attn_output, gate, expected = make_case(SHAPES[0])
    compiled = torch.compile(
        fused_sigmoid_mul,
        dynamic=False,
        fullgraph=True,
    )
    output = compiled(attn_output, gate, inplace=False)
    torch.cuda.synchronize()
    absolute_error = (
        output.detach().cpu().double() - expected.double()
    ).abs().max().item()
    return {
        "variant": "torch.compile(dynamic=False, fullgraph=True)",
        "returned_without_exception": True,
        "output_matches_direct_reference": bool(
            torch.allclose(
                output.detach().cpu(),
                expected,
                rtol=1.0e-2,
                atol=1.0e-2,
            )
        ),
        "reference_max_abs_error": absolute_error,
        "conclusion": "unsupported on this Torch/Triton stack; negative evidence only, not forced through as a supported path",
    }


def unsupported_complex_result() -> dict[str, object]:
    attn_output, gate, _ = make_case(SHAPES[0])
    complex_attn = torch.complex(attn_output, attn_output)
    complex_gate = torch.complex(gate, gate)
    try:
        fused_sigmoid_mul(complex_attn, complex_gate, inplace=False)
    except Exception as error:
        return {
            "variant": "torch.complex64 direct Triton inputs",
            "error": f"{type(error).__name__}: {error}",
            "failed_clearly": True,
        }
    raise AssertionError("complex dtype unexpectedly succeeded")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("results.json")),
    )
    arguments = parser.parse_args()

    torch.cuda.init()
    device = torch.cuda.get_device_properties(0)
    cache_before = cache_size()
    rows = [run_shape(shape) for shape in SHAPES]
    cache_after = cache_size()

    attn_output, gate, _ = make_case(SHAPES[0])
    direct_dispatch = profiler_events(
        lambda: fused_sigmoid_mul(attn_output, gate, inplace=False)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fused_sigmoid_mul(attn_output, gate, inplace=False)
    graph_dispatch = profiler_events(graph.replay)

    report = {
        "label": "delivery-checkout gfx942 fused-sigmoid-mul compiled-path reuse",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "/opt/venv/bin/python reports/j-ca0f72e54fb1/run_reuse_experiment.py",
        "campaign": "repo-e2e-20260909",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_path": module_path("sglang"),
        "kernel_source_path": str(
            Path(__file__).parents[2]
            / "python/sglang/kernels/ops/elementwise/elementwise.py"
        ),
        "environment": {
            "python": "/opt/venv/bin/python",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_path": torch.__file__,
            "torch_cuda_path": module_path("torch.cuda"),
            "hip_version": torch.version.hip,
            "triton_version": importlib.metadata.version("triton"),
            "triton_path": module_path("triton"),
            "sglang_version": importlib.metadata.version("sglang"),
            "sglang_path": module_path("sglang"),
            "gpu_name": device.name,
            "gpu_gcn_arch_name": device.gcnArchName,
            "gpu_uuid": str(device.uuid),
            "gpu_total_memory_bytes": device.total_memory,
            "device_count": torch.cuda.device_count(),
            "hostname": platform.node(),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "image_identity_requested": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "image_local_id_requested": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
            "native_modules": {
                "libtorch_hip": "/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so",
                "libamdhip64": "/opt/rocm/lib/libamdhip64.so.7",
                "libroctracer64": "/opt/rocm/lib/libroctracer64.so.4",
                "libhsa_runtime64": "/opt/rocm/lib/libhsa-runtime64.so.1",
            },
        },
        "operation": "sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul",
        "shape_sequence": [list(shape) for shape in SHAPES],
        "dtype_contract": "float16 inputs and float16 output",
        "reference_method": "independent CPU float64 attn * sigmoid(gate), cast to float16",
        "timing_method": "synchronized perf_counter around direct calls; CUDA/HIP events around graph replays",
        "warm_measurements_per_shape": WARM_MEASUREMENTS,
        "graph_replays_per_shape": GRAPH_REPLAYS,
        "triton_cache_delta": cache_after - cache_before,
        "results": rows,
        "native_dispatch": {
            "direct": direct_dispatch,
            "graph_replay": graph_dispatch,
        },
        "unsupported_variants": [
            unsupported_torch_compile_result(),
            unsupported_complex_result(),
        ],
        "limitations": [
            "one assigned MI300X gfx942 only",
            "bounded operator-level timing, not an end-to-end benchmark",
            "torch.compile is recorded as unsupported negative evidence on this stack",
            "no full model weights, toolchain replacement, or unbounded stress test",
        ],
    }
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
