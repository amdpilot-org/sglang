import importlib.util
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import aiter
import aiter.mla as aiter_mla
import aiter.ops.flydsl as aiter_flydsl
import flydsl
import flydsl.compiler as flyc
import torch
from torch import nn


START = time.monotonic()
OUTPUT_PATH = Path(__file__).with_name("results.json")
NUM_TILES = 4
NUM_SPLITS = 4
HEADS = 16
HEAD_DIM = 512
PARTIAL_ROWS = NUM_TILES * NUM_SPLITS


def module_path(name):
    spec = importlib.util.find_spec(name)
    return spec.origin if spec else None


def git_commit(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def capture_call(function):
    try:
        value = function()
    except Exception as error:
        return {
            "ok": False,
            "exception_type": f"{type(error).__module__}.{type(error).__name__}",
            "message": str(error),
            "traceback": traceback.format_exc(limit=8),
        }
    return {"ok": True, "result_type": type(value).__module__ + "." + type(value).__name__}


def capture_call_in_inference_mode(function):
    def call():
        with torch.inference_mode():
            return function()

    return capture_call(call)


def build_metadata(device):
    reduce_indptr = torch.arange(
        0, PARTIAL_ROWS + 1, NUM_SPLITS, dtype=torch.int32, device=device
    )
    reduce_final_map = torch.stack(
        (
            torch.arange(NUM_TILES, dtype=torch.int32, device=device),
            torch.arange(1, NUM_TILES + 1, dtype=torch.int32, device=device),
        ),
        dim=1,
    ).contiguous()
    reduce_partial_map = torch.arange(PARTIAL_ROWS, dtype=torch.int32, device=device)
    return reduce_indptr, reduce_final_map, reduce_partial_map


def reference(partial_output, partial_lse):
    output = partial_output.view(NUM_TILES, NUM_SPLITS, HEADS, HEAD_DIM).double()
    lse = partial_lse.view(NUM_TILES, NUM_SPLITS, HEADS).double()
    max_lse = lse.max(dim=1, keepdim=True).values
    weights = torch.exp(lse - max_lse)
    denominator = weights.sum(dim=1)
    numerator = (weights.unsqueeze(-1) * output).sum(dim=1)
    expected_output = (numerator / denominator.unsqueeze(-1)).to(torch.bfloat16)
    expected_lse = (max_lse.squeeze(1) + torch.log(denominator)).float()
    return expected_output, expected_lse


def timed_calls(call, warmup=3, repeats=10):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies_ms = []
    for _ in range(repeats):
        start_event.record()
        call()
        end_event.record()
        end_event.synchronize()
        latencies_ms.append(start_event.elapsed_time(end_event))
    return latencies_ms


def main():
    os.environ["AITER_MLA_REDUCE_FLYDSL"] = "1"
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)
    source_partial_output = torch.randn(
        PARTIAL_ROWS, 1, HEADS, HEAD_DIM,
        dtype=torch.float32, device=device, generator=generator,
    )
    source_partial_lse = torch.randn(
        PARTIAL_ROWS, 1, HEADS, 1,
        dtype=torch.float32, device=device, generator=generator,
    ) * 2.0
    source_indptr, source_final_map, source_partial_map = build_metadata(device)

    original_flydsl_reduce = aiter_flydsl.flydsl_mla_reduce_v1
    flydsl_calls = 0

    def counted_flydsl_reduce(*args, **kwargs):
        nonlocal flydsl_calls
        flydsl_calls += 1
        return original_flydsl_reduce(*args, **kwargs)

    aiter_flydsl.flydsl_mla_reduce_v1 = counted_flydsl_reduce
    cases = {}
    first_execution_elapsed_s = None
    try:
        def run_dispatch(partial_output, partial_lse, final_output, final_lse):
            aiter_mla._mla_decode_reduce_v1_dispatch(
                partial_output,
                partial_lse,
                source_indptr,
                source_final_map,
                source_partial_map,
                1,
                0,
                final_output,
                final_lse,
            )

        def run_case(name, partial_output, partial_lse):
            nonlocal first_execution_elapsed_s
            with torch.inference_mode():
                final_output = torch.empty(
                    NUM_TILES, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
                )
                final_lse = torch.empty(
                    NUM_TILES, HEADS, dtype=torch.float32, device=device
                )
                run_dispatch(partial_output, partial_lse, final_output, final_lse)
                torch.cuda.synchronize()
                if first_execution_elapsed_s is None:
                    first_execution_elapsed_s = time.monotonic() - START
                latencies_ms = timed_calls(
                    lambda: run_dispatch(
                        partial_output, partial_lse, final_output, final_lse
                    )
                )
                expected_output, expected_lse = reference(
                    partial_output.detach(), partial_lse.detach()
                )
                output_error = (
                    final_output.float() - expected_output.float()
                ).abs()
                lse_error = (final_lse - expected_lse).abs()
                cases[name] = {
                    "input_requires_grad": bool(partial_output.requires_grad),
                    "input_is_parameter": isinstance(partial_output, nn.Parameter),
                    "input_is_view": bool(partial_output._base is not None),
                    "latency_ms": latencies_ms,
                    "mean_latency_ms": sum(latencies_ms) / len(latencies_ms),
                    "output_max_abs": output_error.max().item(),
                    "output_mean_abs": output_error.mean().item(),
                    "lse_max_abs": lse_error.max().item(),
                    "passed_gates": (
                        output_error.max().item() <= 0.063
                        and lse_error.max().item() <= 0.001
                    ),
                }
                return final_output, final_lse

        with torch.inference_mode():
            ordinary_output = source_partial_output.clone()
            ordinary_lse = source_partial_lse.clone()
        ordinary_result = run_case("ordinary_inference_tensors", ordinary_output, ordinary_lse)

        parameter_output = nn.Parameter(source_partial_output.detach().clone(), requires_grad=True)
        parameter_lse = nn.Parameter(source_partial_lse.detach().clone(), requires_grad=True)
        parameter_result = run_case("requires_grad_parameters", parameter_output, parameter_lse)

        output_base = nn.Parameter(
            torch.randn(
                PARTIAL_ROWS * HEADS * HEAD_DIM + 1,
                dtype=torch.float32,
                device=device,
                generator=generator,
            ),
            requires_grad=True,
        )
        lse_base = nn.Parameter(
            torch.randn(
                PARTIAL_ROWS * HEADS + 1,
                dtype=torch.float32,
                device=device,
                generator=generator,
            ),
            requires_grad=True,
        )
        with torch.no_grad():
            output_view = output_base[:-1].view(PARTIAL_ROWS, 1, HEADS, HEAD_DIM)
            lse_view = lse_base[:-1].view(PARTIAL_ROWS, 1, HEADS, 1)
            output_view.copy_(source_partial_output)
            lse_view.copy_(source_partial_lse)
        view_result = run_case("requires_grad_parameter_views", output_view, lse_view)

        cases["requires_grad_parameters"]["bitwise_equal_to_ordinary"] = bool(
            torch.equal(parameter_result[0], ordinary_result[0])
            and torch.equal(parameter_result[1], ordinary_result[1])
        )
        cases["requires_grad_parameter_views"]["bitwise_equal_to_ordinary"] = bool(
            torch.equal(view_result[0], ordinary_result[0])
            and torch.equal(view_result[1], ordinary_result[1])
        )
    finally:
        aiter_flydsl.flydsl_mla_reduce_v1 = original_flydsl_reduce

    parameter = parameter_output
    parameter_view = output_view
    dlpack_controls = {
        "torch___dlpack___parameter_normal_mode": capture_call(lambda: parameter.__dlpack__()),
        "torch___dlpack___parameter_inference_mode": capture_call_in_inference_mode(
            lambda: parameter.__dlpack__()
        ),
        "flydsl_from_dlpack_parameter_normal_mode": capture_call(
            lambda: flyc.from_dlpack(parameter)
        ),
        "flydsl_from_dlpack_parameter_inference_mode": capture_call_in_inference_mode(
            lambda: flyc.from_dlpack(parameter)
        ),
        "flydsl_from_dlpack_parameter_view_inference_mode": capture_call_in_inference_mode(
            lambda: flyc.from_dlpack(parameter_view)
        ),
        "flydsl_from_dlpack_detached_parameter_inference_mode": capture_call_in_inference_mode(
            lambda: flyc.from_dlpack(parameter.detach())
        ),
    }

    gpu_properties = torch.cuda.get_device_properties(0)
    result = {
        "scope": "gfx942 inference bridge control; not a gfx950 training/autograd claim",
        "operator": "aiter.mla._mla_decode_reduce_v1_dispatch -> aiter.ops.flydsl.flydsl_mla_reduce_v1",
        "shape": {
            "num_tiles": NUM_TILES,
            "num_splits": NUM_SPLITS,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "dtype": "bfloat16",
        },
        "command": "/opt/venv/bin/python reports/j-15d35939bb4c/inference_bridge_control.py",
        "first_gpu_execution_elapsed_s": first_execution_elapsed_s,
        "flydsl_dispatch_calls_observed": flydsl_calls,
        "dispatch_enabled": aiter_mla._flydsl_mla_reduce_enabled(),
        "dispatch_supported": aiter_mla._flydsl_mla_reduce_supported(
            torch.randn(1, 1, HEADS, HEAD_DIM, device=device),
            torch.randn(1, 1, HEADS, 1, device=device),
            torch.empty(1, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device),
            1,
            0,
        ),
        "cases": cases,
        "dlpack_controls": dlpack_controls,
        "timing_method": {
            "warmup_runs": 3,
            "timed_runs": 10,
            "events": "torch.cuda.Event(enable_timing=True)",
            "synchronization": "end_event.synchronize() after each timed run",
        },
        "environment": {
            "python": "/opt/venv/bin/python",
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "torch_path": torch.__file__,
            "aiter_source_path": aiter.__file__,
            "aiter_commit": git_commit("/sgl-workspace/aiter"),
            "aiter_core_native_path": module_path("aiter.jit.module_aiter_core"),
            "flydsl_version": getattr(flydsl, "__version__", None),
            "flydsl_path": flydsl.__file__,
            "delivery_repo_commit": git_commit("/job/sglang"),
            "gpu_name": gpu_properties.name,
            "gpu_arch": gpu_properties.gcnArchName,
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_uuid": str(getattr(gpu_properties, "uuid", None)),
            "qualified_image": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "qualified_local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        },
        "conclusion": {
            "integration_defect_demonstrated": not all(
                case["passed_gates"] for case in cases.values()
            ),
            "candidate_tested": False,
            "candidate_reason": "No gfx942 integration defect was demonstrated; AITER PR 5327 targets the gfx950 DLPack GEMM path.",
        },
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
