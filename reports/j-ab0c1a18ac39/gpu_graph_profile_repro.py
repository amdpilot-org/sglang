"""Minimal real ROCm graph/profiler reproduction for issue 39092."""

import contextlib
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.runner import decode_cuda_graph_runner as runner_mod
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)


class _NoopPrecarve:
    measure = contextlib.nullcontext

    @staticmethod
    def mint():
        return None


def make_backend(runner):
    backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._pool = None
    backend._capture_stream = None
    backend._precarve = _NoopPrecarve()
    backend._reuse_output_buffer = False
    backend._output_buffer = None
    backend._memory_saver_adapter = None
    backend._cuda_graph_runner = runner
    backend._device_module = torch.cuda
    backend._tp_group = SimpleNamespace(barrier=lambda: None)
    return backend


def main():
    output_base = Path(__file__).parent / "raw" / "gpu_profile"
    output_base.mkdir(parents=True, exist_ok=True)
    for old_trace in (output_base / "graph_capture_profile").glob("*.json.gz"):
        old_trace.unlink()
    os.environ["SGLANG_GRAPH_BATCH_CAPTURE"] = "1"
    os.environ.pop("SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE", None)
    os.environ["SGLANG_TORCH_PROFILER_DIR"] = str(output_base)

    runner = SimpleNamespace(capture_bs=[7, 11], enable_profile_cuda_graph=True)
    runner._graph_batch_capture_active = (
        DecodeCudaGraphRunner._graph_batch_capture_active.__get__(runner)
    )
    runner._enqueue_profile_capture_identity = (
        DecodeCudaGraphRunner._enqueue_profile_capture_identity.__get__(runner)
    )
    with (
        mock.patch.object(
            runner_mod, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ),
        mock.patch("torch.cuda.memory._record_memory_history"),
    ):
        profiler = DecodeCudaGraphRunner._init_profile_context_and_memory_record(runner)

    backend = make_backend(runner)
    runner._profiler = profiler
    identities = [
        ShapeKey(bs, stream, lora, attention)
        for stream in (0, 1)
        for bs in (11, 7)
        for lora in ("lora", "nolora")
        for attention in ("dense", "sparse")
    ]
    eager_functions = {}
    static_inputs = {}
    for stream in (0, 1):
        # DecodeCudaGraphRunner.capture() enters the same profiler separately
        # for every PDMUX stream group.
        with profiler:
            for index, key in enumerate(identities):
                if key.stream_idx != stream:
                    continue
                x = torch.arange(key.size, device="cuda", dtype=torch.float32)
                scale = float(index + 1)
                static_inputs[key] = x

                def forward(x=x, scale=scale, key=key):
                    y = x * scale + 3.0
                    if key.attention_variant == "sparse":
                        y = torch.sin(y) + y
                    if key.variant_label == "nolora":
                        y = y.square()
                    return y

                eager_functions[key] = forward
                backend.capture_one(key, forward)

    torch.cuda.synchronize()
    max_abs_error = 0.0
    for key in identities:
        static_inputs[key].add_(1.0)
        reference = eager_functions[key]().clone()
        backend._graphs[key].replay()
        torch.cuda.synchronize()
        max_abs_error = max(
            max_abs_error,
            (backend._outputs[key] - reference).abs().max().item(),
        )

    trace_dir = output_base / "graph_capture_profile"
    traces = sorted(trace_dir.glob("*.json.gz"))
    trace_evidence = []
    for path in traces:
        with gzip.open(path, "rt") as f:
            payload = json.load(f)
        names = [event.get("name", "") for event in payload["traceEvents"]]
        steps = sorted(name for name in names if name.startswith("ProfilerStep#"))
        kernels = sum(
            event.get("cat") in {"kernel", "Kernel"}
            or "kernel" in event.get("cat", "").lower()
            for event in payload["traceEvents"]
        )
        trace_evidence.append(
            {"file": path.name, "profiler_steps": steps, "kernel_events": kernels}
        )

    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "capture_count": len(identities),
        "trace_count": len(traces),
        "unique_trace_names": len({path.name for path in traces}),
        "max_abs_error_after_replay": max_abs_error,
        "traces": trace_evidence,
    }
    print(json.dumps(result, indent=2))
    if len(traces) != len(identities):
        raise AssertionError(result)
    if len({path.name for path in traces}) != len(identities):
        raise AssertionError(result)
    if max_abs_error != 0.0:
        raise AssertionError(result)


if __name__ == "__main__":
    main()
