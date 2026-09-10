"""Bounded gfx942 validation for ROCm router GEMM execution representations."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

started = time.perf_counter()

import sglang
import sgl_kernel
import torch
from aiter import tuned_gemm
from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm


NUM_EXPERTS = 256
HIDDEN_DIM = 7168
ROW_PADDING = 8
WARMUP_ITERATIONS = 3
TIMED_ITERATIONS = 10


def guarded_view(rows, columns, padding, fill):
    guard = torch.full((16,), fill, device="cuda", dtype=torch.bfloat16)
    storage = torch.empty(
        guard.numel() + rows * (columns + padding) + guard.numel(),
        device="cuda",
        dtype=torch.bfloat16,
    )
    storage[: guard.numel()] = guard
    storage[-guard.numel() :] = guard
    base = storage[guard.numel() : -guard.numel()].view(
        rows, columns + padding
    )
    base.copy_(torch.randn_like(base))
    return guard, storage, base[:, :columns]


def mean_microseconds(function):
    for _ in range(WARMUP_ITERATIONS):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERATIONS):
        function()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) / TIMED_ITERATIONS * 1000, 3)


def dispatch_config(num_tokens):
    config = get_GEMM_A16W16_config(
        M=num_tokens,
        N=NUM_EXPERTS,
        K=HIDDEN_DIM,
        bias=False,
        dtype=str(torch.bfloat16),
        otype=str(torch.bfloat16),
        scaleAB=False,
        bpreshuffle=False,
    )
    return {"libtype": config["libtype"], "solidx": config["solidx"]}


assert torch.cuda.is_available() and torch.cuda.device_count() == 1
device_properties = torch.cuda.get_device_properties(0)
results = []

for representation in ("contiguous", "row_sliced"):
    padding = 0 if representation == "contiguous" else ROW_PADDING
    for num_tokens in (1, 8, 16):
        torch.manual_seed(38695 + num_tokens)
        hidden_guard, hidden_storage, hidden_states = guarded_view(
            num_tokens, HIDDEN_DIM, padding, 123.25
        )
        weight_guard, weight_storage, weight = guarded_view(
            NUM_EXPERTS, HIDDEN_DIM, 0, 456.25
        )
        reference = (
            hidden_states.double().cpu() @ weight.double().cpu().T
        ).to(torch.bfloat16)
        output = aiter_dsv3_router_gemm(hidden_states, weight)
        torch.cuda.synchronize()
        max_abs_error = (
            output.float() - reference.float().cuda()
        ).abs().max().item()
        torch.testing.assert_close(
            output.float(),
            reference.float().cuda(),
            atol=1.0,
            rtol=1e-2,
        )
        guards_unchanged = all(
            (
                torch.equal(hidden_storage[:16], hidden_guard),
                torch.equal(hidden_storage[-16:], hidden_guard),
                torch.equal(weight_storage[:16], weight_guard),
                torch.equal(weight_storage[-16:], weight_guard),
            )
        )
        assert guards_unchanged
        timing = mean_microseconds(
            lambda: aiter_dsv3_router_gemm(hidden_states, weight)
        )
        results.append(
            {
                "representation": representation,
                "num_tokens": num_tokens,
                "hidden_states_stride": list(hidden_states.stride()),
                "pytorch_is_contiguous": hidden_states.is_contiguous(),
                "output_dtype": str(output.dtype),
                "max_abs_error_vs_cpu_float64_rounded_to_bf16": max_abs_error,
                "input_sentinel_guards_unchanged": guards_unchanged,
                "microseconds_mean_10_iterations": timing,
                "aiter_dispatch": dispatch_config(num_tokens),
            }
        )

unsupported_results = []
hidden_states = torch.randn(
    1, HIDDEN_DIM * 2, device="cuda", dtype=torch.bfloat16
)[:, ::2]
weight = torch.randn(
    NUM_EXPERTS, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16
)
try:
    aiter_dsv3_router_gemm(hidden_states, weight)
    raise AssertionError("inner-strided hidden states were unexpectedly accepted")
except RuntimeError as error:
    unsupported_results.append(
        {
            "variant": "inner_strided_hidden_states",
            "stride": list(hidden_states.stride()),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )

hidden_states = torch.randn(1, HIDDEN_DIM, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(
    NUM_EXPERTS,
    HIDDEN_DIM + ROW_PADDING,
    device="cuda",
    dtype=torch.bfloat16,
)[:, :HIDDEN_DIM]
try:
    aiter_dsv3_router_gemm(hidden_states, weight)
    raise AssertionError("non-contiguous weight was unexpectedly accepted")
except RuntimeError as error:
    unsupported_results.append(
        {
            "variant": "row_sliced_non_contiguous_weight",
            "stride": list(weight.stride()),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )

record = {
    "label": "mirror-checkout candidate validation",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "first_gpu_execution_elapsed_seconds": round(
        time.perf_counter() - started, 3
    ),
    "gpu": {
        "name": device_properties.name,
        "gcn_arch_name": device_properties.gcnArchName,
        "compute_units": device_properties.multi_processor_count,
        "total_memory_bytes": device_properties.total_memory,
    },
    "image": {
        "name": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
        "local_image_id": "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
        "identity_source": "operator-provided local image ID; not a registry digest",
    },
    "python": sys.executable,
    "torch": {
        "version": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
    },
    "git_commit": subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_paths": {
        "sglang": os.path.dirname(inspect.getfile(sglang)),
        "sgl_kernel": os.path.dirname(inspect.getfile(sgl_kernel)),
        "router_helper": inspect.getfile(aiter_dsv3_router_gemm),
        "aiter_tgemm": inspect.getfile(tuned_gemm),
    },
    "native_paths": {
        "aiter_core": "/sgl-workspace/aiter/aiter/jit/module_aiter_core.so",
        "aiter_custom": "/sgl-workspace/aiter/aiter/jit/module_custom.so",
    },
    "operation": "aiter_dsv3_router_gemm(hidden_states, weight)",
    "reference": "independent CPU float64 matmul rounded to bf16",
    "numerical_gate": {"atol": 1.0, "rtol": 1e-2, "output_dtype": "hidden_states.dtype"},
    "sentinel_storage": "16-element guards before and after hidden states and weight",
    "timing": f"CUDA events, {WARMUP_ITERATIONS} warmups, {TIMED_ITERATIONS} timed iterations, mean reported",
    "preserved_contracts": {
        "aliasing": "no input/output alias introduced",
        "static_graph_address": "no allocator or output-buffer change",
        "dtype": "otype remains hidden_states.dtype",
    },
    "results": results,
    "unsupported_results": unsupported_results,
    "command": "/opt/venv/bin/python reports/j-00f7a4657fb9/validate_execution_representations.py",
}

output_path = Path(__file__).with_name("execution-representation-results.json")
output_path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
