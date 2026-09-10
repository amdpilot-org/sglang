"""Verify that HIP graph replay remains numerically correct while preserving
the distinction between synchronized GPU execution order and profiler marker
attribution."""

import json
import sys
import time
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci


register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


STEPS = 5


def _kernel_chain(input_tensor, weight, bias):
    return torch.nn.functional.gelu(torch.addmm(bias, input_tensor, weight))


def _overlaps(event, start, end):
    event_start = event["ts"]
    event_end = event_start + event.get("dur", 0)
    return event_start < end and event_end > start


def _trace_summary(trace_path):
    with open(trace_path, encoding="utf-8") as handle:
        events = json.load(handle)["traceEvents"]

    cpu_markers = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("step[")
    ]
    gpu_markers = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "gpu_user_annotation"
        and str(event.get("name", "")).startswith("step[")
    ]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    attribution_markers = sorted(
        gpu_markers or cpu_markers,
        key=lambda event: event["ts"],
    )
    intervals = [
        (event["ts"], event["ts"] + event.get("dur", 0))
        for event in attribution_markers
    ]
    chain_kernels = [
        event
        for event in kernels
        if any(_overlaps(event, start, end) for start, end in intervals)
    ]
    per_marker = []
    for marker, (start, end) in zip(attribution_markers, intervals):
        marker_kernels = [
            event for event in chain_kernels if _overlaps(event, start, end)
        ]
        per_marker.append(
            {
                "marker": marker["name"],
                "kernel_events": len(marker_kernels),
                "busy_us": sum(
                    max(
                        0.0,
                        min(end, event["ts"] + event.get("dur", 0))
                        - max(start, event["ts"]),
                    )
                    for event in marker_kernels
                ),
            }
        )
    last_cpu_end = (
        max(event["ts"] + event.get("dur", 0) for event in cpu_markers)
        if cpu_markers
        else float("inf")
    )
    return {
        "cpu_marker_count": len(cpu_markers),
        "gpu_marker_count": len(gpu_markers),
        "all_kernel_event_count": len(kernels),
        "chain_kernel_event_count": len(chain_kernels),
        "chain_kernel_events_after_last_cpu_marker": sum(
            event["ts"] >= last_cpu_end for event in chain_kernels
        ),
        "per_marker": per_marker,
    }


def _profile_mode(mode, operation, reference, trace_path):
    event_durations_ms = []
    completion_wall_s = []

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for step in range(STEPS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.profiler.record_function(f"step[{mode} {step}]"):
                output = operation()
            end.record()
            end.synchronize()
            event_durations_ms.append(start.elapsed_time(end))
            completion_wall_s.append(time.perf_counter())

    profiler.export_chrome_trace(str(trace_path))

    torch.cuda.synchronize()
    result = output.detach().cpu()
    return {
        "trace_path": str(trace_path),
        "event_durations_ms": event_durations_ms,
        "completion_wall_s": completion_wall_s,
        "reference_max_abs_diff": float((result - reference).abs().max().item()),
        "reference_allclose": bool(
            torch.allclose(result, reference, atol=1e-5, rtol=1e-5)
        ),
        "trace_summary": _trace_summary(trace_path),
    }


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires an available HIP GPU",
)
def test_hip_graph_replay_attribution_preserves_execution_order(tmp_path):
    torch.manual_seed(31545)
    device = torch.device("cuda:0")
    input_tensor = torch.randn(256, 256, device=device, dtype=torch.float32)
    weight = torch.randn(256, 256, device=device, dtype=torch.float32) * 0.05
    bias = torch.randn(256, device=device, dtype=torch.float32) * 0.05

    for _ in range(3):
        _kernel_chain(input_tensor, weight, bias)
    torch.cuda.synchronize()

    reference = _kernel_chain(input_tensor, weight, bias).detach().cpu()
    static_input = input_tensor.clone()
    static_weight = weight.clone()
    static_bias = bias.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = _kernel_chain(static_input, static_weight, static_bias)

    def replay_graph():
        graph.replay()
        return static_output

    eager = _profile_mode(
        "eager",
        lambda: _kernel_chain(input_tensor, weight, bias),
        reference,
        tmp_path / "eager.json",
    )
    graph_replay = _profile_mode(
        "graph-replay",
        replay_graph,
        reference,
        tmp_path / "graph-replay.json",
    )
    summary = {
        "actual_gpu_ordering": "Each step's end event is synchronized before the next step starts.",
        "eager": eager,
        "graph_replay": graph_replay,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    assert eager["reference_allclose"]
    assert graph_replay["reference_allclose"]
    for result in (eager, graph_replay):
        assert all(duration > 0.0 for duration in result["event_durations_ms"])
        assert result["reference_max_abs_diff"] <= 1e-5
        assert result["completion_wall_s"] == sorted(result["completion_wall_s"])
        assert len(result["completion_wall_s"]) == len(set(result["completion_wall_s"]))
        assert Path(result["trace_path"]).stat().st_size > 0
        assert result["trace_summary"]["cpu_marker_count"] == STEPS

    assert (
        eager["trace_summary"]["chain_kernel_event_count"]
        == graph_replay["trace_summary"]["chain_kernel_event_count"]
        > 0
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
