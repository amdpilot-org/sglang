import json
import os
import platform
import statistics
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel


FP8_DTYPE = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
SENTINEL = -9999.0
OUTPUT_BOUND = 2048.0


def build_case(group_sizes):
    num_experts = len(group_sizes)
    output_size = 128
    input_size = 128
    total_rows = sum(group_sizes)
    block_size_m = 16

    activation_values = torch.tensor(
        [0, 1, -1, 2, -2], device="cuda", dtype=torch.float32
    )
    activation = activation_values[
        torch.arange(total_rows * input_size, device="cuda") % 5
    ].reshape(total_rows, input_size).to(FP8_DTYPE)
    activation_scale = torch.empty((total_rows, 1), device="cuda", dtype=torch.float32)

    weight_values = torch.tensor(
        [1, -1, 2, -2, 0], device="cuda", dtype=torch.float32
    )
    weight = weight_values[
        torch.arange(num_experts * output_size * input_size, device="cuda") % 5
    ].reshape(num_experts, output_size, input_size).to(FP8_DTYPE)
    weight_scale = torch.empty(
        (num_experts, 1, 1), device="cuda", dtype=torch.float32
    )

    for row in range(total_rows):
        activation_scale[row, 0] = 0.25 * (row + 1)
    for expert in range(num_experts):
        weight_scale[expert, 0, 0] = 0.5 * (expert + 1)

    output = torch.full(
        (total_rows, output_size), SENTINEL, device="cuda", dtype=torch.float32
    )
    topk_weights = torch.ones((total_rows, 1), device="cuda", dtype=torch.float32)
    topk_ids = torch.arange(
        total_rows, device="cuda", dtype=torch.int32
    ).reshape(total_rows, 1)

    sorted_token_ids = []
    expert_ids = []
    row_start = 0
    for expert, group_size in enumerate(group_sizes):
        row_end = row_start + group_size
        sorted_token_ids.extend(range(row_start, row_end))
        sorted_token_ids.extend([total_rows] * (block_size_m - group_size))
        expert_ids.append(expert)
        row_start = row_end

    sorted_token_ids = torch.tensor(
        sorted_token_ids, device="cuda", dtype=torch.int32
    )
    expert_ids = torch.tensor(expert_ids, device="cuda", dtype=torch.int32)
    num_tokens_post_padded = torch.tensor(
        [sorted_token_ids.numel()], device="cuda", dtype=torch.int32
    )
    config = {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
    }

    def dispatch():
        invoke_fused_moe_kernel(
            activation,
            weight,
            None,
            output,
            activation_scale,
            weight_scale,
            None,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            1,
            config,
            tl.float32,
            True,
            False,
            False,
            False,
            False,
            [128, 128],
        )
        return output

    reference = torch.empty_like(output)
    row_start = 0
    for expert, group_size in enumerate(group_sizes):
        row_end = row_start + group_size
        if group_size:
            dequantized_activation = (
                activation[row_start:row_end].float()
                * activation_scale[row_start:row_end]
            )
            dequantized_weight = weight[expert].float() * weight_scale[expert]
            reference[row_start:row_end] = torch.mm(
                dequantized_activation, dequantized_weight.T
            )
        row_start = row_end

    return dispatch, reference


def time_case(name, group_sizes, repeats):
    dispatch, reference = build_case(group_sizes)
    torch.cuda.synchronize()
    first_dispatch_start = time.perf_counter()
    output = dispatch()
    torch.cuda.synchronize()
    first_dispatch_ms = (time.perf_counter() - first_dispatch_start) * 1000

    samples_ms = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        dispatch()
        end_event.record()
        torch.cuda.synchronize()
        samples_ms.append(start_event.elapsed_time(end_event))

    max_difference = float((output - reference).abs().max())
    return {
        "name": name,
        "group_sizes": group_sizes,
        "empty_groups": sum(size == 0 for size in group_sizes),
        "one_token_groups": sum(size == 1 for size in group_sizes),
        "output_shape": list(output.shape),
        "activation_scale_shape": [sum(group_sizes), 1],
        "weight_scale_shape": [len(group_sizes), 1, 1],
        "reference_max_abs_difference": max_difference,
        "reference_gate_max_abs_difference": 0.0,
        "sentinel_output_bound": OUTPUT_BOUND,
        "output_max_abs": float(output.abs().max()),
        "output_within_sentinel_bound": bool(output.abs().max() <= OUTPUT_BOUND),
        "sentinel_values_remaining": int((output == SENTINEL).sum()),
        "all_finite": bool(torch.isfinite(output).all()),
        "timing": {
            "first_dispatch_ms": first_dispatch_ms,
            "warm_samples": repeats,
            "warm_mean_ms": statistics.mean(samples_ms),
            "warm_median_ms": statistics.median(samples_ms),
            "warm_min_ms": min(samples_ms),
            "warm_max_ms": max(samples_ms),
        },
    }


def main():
    process_start = time.perf_counter()
    torch.cuda.init()
    first_gpu_execution_elapsed_ms = (time.perf_counter() - process_start) * 1000

    cases = [
        time_case("empty_and_one_token", [0, 1, 2, 3], 20),
        time_case("mixed_small_groups", [0, 1, 0, 2, 1, 0, 3, 1], 20),
    ]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd="/job/sglang", text=True
    ).strip()
    result = {
        "label": "persistent-checkout post-fix MI300X result",
        "campaign": "repo-e2e-20260909",
        "source_commit": source_commit,
        "source_path": (
            "/job/sglang/python/sglang/kernels/ops/moe/"
            "fused_moe_triton_kernels.py"
        ),
        "test_path": (
            "/job/sglang/test/registered/kernels/ops/moe/"
            "test_fused_moe_group_boundaries.py"
        ),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_path": torch.__file__,
        "torch_version": torch.__version__,
        "torch_hip": torch.version.hip,
        "triton_path": triton.__file__,
        "triton_native_path": (
            "/opt/venv/lib/python3.10/site-packages/triton/_C/libtriton.so"
        ),
        "torch_native_path": (
            "/opt/venv/lib/python3.10/site-packages/torch/"
            "_C.cpython-310-x86_64-linux-gnu.so"
        ),
        "installed_sgl_kernel_native_path": (
            "/opt/venv/lib/python3.10/site-packages/sgl_kernel/"
            "common_ops.cpython-310-x86_64-linux-gnu.so"
        ),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "image": {
            "operator_provided_reference": (
                "amdpilotv2/open-job-mi300:"
                "jit-config-readable-260909-banff5"
            ),
            "operator_provided_local_image_id": (
                "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c"
                "63a20b3ccfd63f1"
            ),
            "hostname_is_not_image_identity": True,
            "container_hostname": platform.node(),
        },
        "cache": {
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        },
        "operator": "sglang fused_moe_kernel direct grouped FP8 GEMM",
        "reference_method": "independent float32 dequantized torch.mm per expert group",
        "numerical_gates": {
            "rtol": 0.0,
            "atol": 0.0,
            "sentinel_output_bound": OUTPUT_BOUND,
            "require_all_finite": True,
            "require_no_sentinel_remaining": True,
        },
        "timing_method": "one real dispatch before 20 bounded CUDA-event timed warm dispatches",
        "first_gpu_execution_elapsed_ms": first_gpu_execution_elapsed_ms,
        "cases": cases,
    }

    assert all(case["reference_max_abs_difference"] == 0.0 for case in cases)
    assert all(case["output_within_sentinel_bound"] for case in cases)
    assert all(case["sentinel_values_remaining"] == 0 for case in cases)
    assert all(case["all_finite"] for case in cases)

    output_path = "/job/sglang/reports/j-505b9a648f65/post-patch.json"
    with open(output_path, "w") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
