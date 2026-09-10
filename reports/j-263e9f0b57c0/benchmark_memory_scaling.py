import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as functional


CASES = [
    (1, 1),
    (1, 32),
    (2, 64),
    (4, 128),
    (8, 256),
    (16, 512),
]
HIDDEN_SIZE = 2048
ATTENTION_SIZE = 2048
HEAD_COUNT = 8
HEAD_DIMENSION = ATTENTION_SIZE // HEAD_COUNT
FP8_DTYPE = torch.float8_e4m3fnuz
BF16_DTYPE = torch.bfloat16
INPUT_SCALE_VALUE = 1.0
WEIGHT_SCALE_VALUE = 0.03125
TIMING_REPETITIONS = 10
TIMING_WARMUP = 3
ALLOCATION_LIMIT_BYTES = 48 * 1024**3


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def unique_storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def fp8_linear(input_bf16, weight_fp8, input_scale, weight_scale):
    input_fp8 = input_bf16.to(FP8_DTYPE)
    output_bf16 = torch._scaled_mm(
        input_fp8,
        weight_fp8.t(),
        scale_a=input_scale,
        scale_b=weight_scale,
        out_dtype=BF16_DTYPE,
    )
    return output_bf16, input_fp8


def block_forward(input_bf16, k_cache, v_cache, qkv_weight, output_weight, input_scale, qkv_scale, output_scale):
    batch_size, context_length, _ = input_bf16.shape
    flattened_length = batch_size * context_length
    qkv_output, input_fp8 = fp8_linear(
        input_bf16.reshape(flattened_length, HIDDEN_SIZE),
        qkv_weight,
        input_scale,
        qkv_scale,
    )
    query, key, value = qkv_output.split(ATTENTION_SIZE, dim=-1)
    query = query.view(batch_size, context_length, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    key = key.view(batch_size, context_length, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    value = value.view(batch_size, context_length, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    k_cache.copy_(key.transpose(1, 2).reshape(batch_size, context_length, ATTENTION_SIZE))
    v_cache.copy_(value.transpose(1, 2).reshape(batch_size, context_length, ATTENTION_SIZE))
    k_cache_view = k_cache.view(batch_size, context_length, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    v_cache_view = v_cache.view(batch_size, context_length, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    attention_output = functional.scaled_dot_product_attention(query, k_cache_view, v_cache_view)
    attention_flat = attention_output.transpose(1, 2).contiguous().view(flattened_length, ATTENTION_SIZE)
    block_output, attention_fp8 = fp8_linear(
        attention_flat,
        output_weight,
        input_scale,
        output_scale,
    )
    return {
        "input_fp8": input_fp8,
        "qkv_output": qkv_output,
        "attention_output": attention_output,
        "attention_flat": attention_flat,
        "attention_fp8": attention_fp8,
        "block_output": block_output,
    }


def reference_forward(input_bf16, input_fp8, qkv_weight, output_weight, input_scale, qkv_scale, output_scale):
    batch_size, context_length, _ = input_bf16.shape
    flattened_length = batch_size * context_length
    reference_qkv = (input_fp8.to(BF16_DTYPE) * input_scale.item()) @ (
        qkv_weight.to(BF16_DTYPE).t() * qkv_scale.item()
    )
    reference_query, reference_key, reference_value = reference_qkv.split(ATTENTION_SIZE, dim=-1)
    reference_query = reference_query.view(batch_size, context_length, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    reference_key = reference_key.view(batch_size, context_length, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    reference_value = reference_value.view(batch_size, context_length, HEAD_COUNT, HEAD_DIMENSION).transpose(1, 2)
    reference_scores = torch.matmul(reference_query, reference_key.transpose(-1, -2)) / math.sqrt(HEAD_DIMENSION)
    reference_probabilities = torch.softmax(reference_scores, dim=-1)
    reference_attention = torch.matmul(reference_probabilities, reference_value)
    reference_attention_flat = reference_attention.transpose(1, 2).contiguous().view(flattened_length, ATTENTION_SIZE)
    reference_attention_fp8 = reference_attention_flat.to(FP8_DTYPE)
    reference_output = (reference_attention_fp8.to(BF16_DTYPE) * input_scale.item()) @ (
        output_weight.to(BF16_DTYPE).t() * output_scale.item()
    )
    return reference_output


def timing_samples(input_bf16, k_cache, v_cache, qkv_weight, output_weight, input_scale, qkv_scale, output_scale):
    for _ in range(TIMING_WARMUP):
        block_forward(input_bf16, k_cache, v_cache, qkv_weight, output_weight, input_scale, qkv_scale, output_scale)
    torch.cuda.synchronize()
    samples = []
    for _ in range(TIMING_REPETITIONS):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        block_forward(input_bf16, k_cache, v_cache, qkv_weight, output_weight, input_scale, qkv_scale, output_scale)
        end_event.record()
        torch.cuda.synchronize()
        samples.append(start_event.elapsed_time(end_event) * 1e-3)
    return samples


def main():
    torch.manual_seed(263)
    device = torch.device("cuda:0")
    qkv_weight = torch.randn(3 * ATTENTION_SIZE, HIDDEN_SIZE, device=device, dtype=BF16_DTYPE).to(FP8_DTYPE)
    output_weight = torch.randn(HIDDEN_SIZE, ATTENTION_SIZE, device=device, dtype=BF16_DTYPE).to(FP8_DTYPE)
    input_scale = torch.tensor([INPUT_SCALE_VALUE], device=device, dtype=torch.float32)
    qkv_scale = torch.tensor([WEIGHT_SCALE_VALUE], device=device, dtype=torch.float32)
    output_scale = torch.tensor([WEIGHT_SCALE_VALUE], device=device, dtype=torch.float32)
    weight_baseline = torch.cuda.memory_allocated()
    warmup_input = torch.zeros(1, HIDDEN_SIZE, device=device, dtype=BF16_DTYPE).to(FP8_DTYPE)
    warmup_output = torch._scaled_mm(
        warmup_input,
        qkv_weight.t(),
        scale_a=input_scale,
        scale_b=qkv_scale,
        out_dtype=BF16_DTYPE,
    )
    torch.cuda.synchronize()
    del warmup_input, warmup_output
    native_gemm_workspace_bytes = torch.cuda.memory_allocated() - weight_baseline
    weight_and_native_baseline = torch.cuda.memory_allocated()
    results = []
    for batch_size, context_length in CASES:
        flattened_length = batch_size * context_length
        input_bf16 = torch.randn(batch_size, context_length, HIDDEN_SIZE, device=device, dtype=BF16_DTYPE)
        k_cache = torch.empty(batch_size, context_length, ATTENTION_SIZE, device=device, dtype=BF16_DTYPE)
        v_cache = torch.empty(batch_size, context_length, ATTENTION_SIZE, device=device, dtype=BF16_DTYPE)
        first_start = time.monotonic()
        warmup_actual = block_forward(
            input_bf16,
            k_cache,
            v_cache,
            qkv_weight,
            output_weight,
            input_scale,
            qkv_scale,
            output_scale,
        )
        torch.cuda.synchronize()
        first_elapsed = time.monotonic() - first_start
        del warmup_actual, input_bf16, k_cache, v_cache
        torch.cuda.empty_cache()
        case_baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        input_bf16 = torch.randn(batch_size, context_length, HIDDEN_SIZE, device=device, dtype=BF16_DTYPE)
        k_cache = torch.empty(batch_size, context_length, ATTENTION_SIZE, device=device, dtype=BF16_DTYPE)
        v_cache = torch.empty(batch_size, context_length, ATTENTION_SIZE, device=device, dtype=BF16_DTYPE)
        actual = block_forward(
            input_bf16,
            k_cache,
            v_cache,
            qkv_weight,
            output_weight,
            input_scale,
            qkv_scale,
            output_scale,
        )
        torch.cuda.synchronize()
        live_workspace_bytes = torch.cuda.memory_allocated() - case_baseline
        peak_workspace_bytes = torch.cuda.max_memory_allocated() - case_baseline
        reference_output = reference_forward(
            input_bf16,
            actual["input_fp8"],
            qkv_weight,
            output_weight,
            input_scale,
            qkv_scale,
            output_scale,
        )
        output_difference = actual["block_output"].float() - reference_output.float()
        relative_l2 = (output_difference.norm() / reference_output.float().norm()).item()
        maximum_absolute_error = output_difference.abs().max().item()
        mean_absolute_error = output_difference.abs().mean().item()
        torch.testing.assert_close(
            actual["block_output"],
            reference_output,
            rtol=0.02,
            atol=1.0,
        )
        assert torch.isfinite(actual["block_output"]).all()
        assert actual["block_output"].shape == (flattened_length, HIDDEN_SIZE)
        samples = timing_samples(
            input_bf16,
            k_cache,
            v_cache,
            qkv_weight,
            output_weight,
            input_scale,
            qkv_scale,
            output_scale,
        )
        predicted_tensor_workspace_bytes = unique_storage_bytes(
            [
                input_bf16,
                actual["input_fp8"],
                actual["qkv_output"],
                actual["attention_output"],
                actual["attention_flat"],
                actual["attention_fp8"],
                actual["block_output"],
            ]
        )
        predicted_kv_workspace_bytes = unique_storage_bytes([k_cache, v_cache])
        results.append(
            {
                "batch_size": batch_size,
                "context_length": context_length,
                "flattened_m": flattened_length,
                "predicted_tensor_workspace_bytes": predicted_tensor_workspace_bytes,
                "predicted_kv_workspace_bytes": predicted_kv_workspace_bytes,
                "predicted_total_workspace_bytes": predicted_tensor_workspace_bytes + predicted_kv_workspace_bytes,
                "measured_live_tensor_kv_workspace_bytes": live_workspace_bytes,
                "measured_peak_tensor_kv_workspace_bytes": peak_workspace_bytes,
                "live_minus_predicted_bytes": live_workspace_bytes - predicted_tensor_workspace_bytes - predicted_kv_workspace_bytes,
                "numerical": {
                    "maximum_absolute_error": maximum_absolute_error,
                    "mean_absolute_error": mean_absolute_error,
                    "relative_l2_error": relative_l2,
                    "gate": "torch.testing.assert_close rtol=0.02 atol=1.0; finite output; exact shape",
                },
                "timing": {
                    "method": f"{TIMING_REPETITIONS} CUDA event samples after {TIMING_WARMUP} warmups",
                    "first_forward_elapsed_seconds": first_elapsed,
                    "median_seconds": statistics.median(samples),
                    "mean_seconds": statistics.mean(samples),
                    "minimum_seconds": min(samples),
                    "maximum_seconds": max(samples),
                    "raw_seconds": samples,
                },
            }
        )
        del reference_output, actual, input_bf16, k_cache, v_cache
        torch.cuda.empty_cache()
    maximum_live = max(item["measured_live_tensor_kv_workspace_bytes"] for item in results)
    maximum_peak = max(item["measured_peak_tensor_kv_workspace_bytes"] for item in results)
    assert maximum_peak < ALLOCATION_LIMIT_BYTES
    report = {
        "label": "bounded MI300X FP8 block memory-scaling evidence",
        "source_commit": git_commit(),
        "source_path": "/job/sglang",
        "python_path": "/opt/venv/bin/python",
        "torch_path": torch.__file__,
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "native_torch_library": "/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "architecture": "gfx942",
            "capability": torch.cuda.get_device_capability(0),
            "count": torch.cuda.device_count(),
            "total_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "image_identity": {
            "operator_provided_image": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "operator_provided_local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        },
        "kernel": "torch._scaled_mm with float8_e4m3fnuz inputs and BF16 output",
        "unsupported_installed_backend": {
            "name": "sgl_kernel.fp8_scaled_mm",
            "error": "AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'fp8_scaled_mm'",
        },
        "weight_storage": {
            "qkv_weight_fp8_bytes": tensor_bytes(qkv_weight),
            "output_weight_fp8_bytes": tensor_bytes(output_weight),
            "total_fp8_weight_bytes": tensor_bytes(qkv_weight) + tensor_bytes(output_weight),
            "under_4gb": tensor_bytes(qkv_weight) + tensor_bytes(output_weight) < 4 * 1024**3,
        },
        "native_gemm_workspace_bytes": native_gemm_workspace_bytes,
        "native_gemm_workspace_note": "Persistent Torch/ROCm GEMM workspace is measured separately and excluded from tensor/KV workspace comparisons.",
        "case_count": len(CASES),
        "allocation_limit_bytes": ALLOCATION_LIMIT_BYTES,
        "maximum_measured_live_tensor_kv_workspace_bytes": maximum_live,
        "maximum_measured_peak_tensor_kv_workspace_bytes": maximum_peak,
        "all_cases_under_48gb": maximum_peak < ALLOCATION_LIMIT_BYTES,
        "results": results,
    }
    output_path = Path(__file__).with_name("results.json")
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
