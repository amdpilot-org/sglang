import json
import math
import subprocess
import sys
import time

import torch

sys.path.insert(0, "/job/sglang/python")

from sglang.kernels.ops.attention.merge_state import merge_state_triton


DEVICE = "cuda:0"
QUERY_TOKENS = 8
QUERY_HEADS = 4
HEAD_DIM = 128
PREFIX_KEYS = 64
SCALE = 1.0 / math.sqrt(HEAD_DIM)
TAIL_CASES = (0, 1, 3, 7)


def adversarial(shape, offset, modulus, spread):
    count = torch.tensor(shape).prod().item()
    base = torch.linspace(-spread, spread, count, device=DEVICE, dtype=torch.float32)
    pattern = (
        torch.arange(offset, count + offset, device=DEVICE, dtype=torch.float32) % modulus
    ).sub(modulus // 2).div(modulus // 2)
    return (base * pattern).reshape(shape)


def partial_state(query, key, value):
    scores = torch.einsum("thd,shd->hts", query.float(), key.float()) * SCALE
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hts,shd->thd", probabilities, value.float())
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1).contiguous()
    return output.contiguous(), lse


def bounded_timing(function, warmup=3, iterations=20):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    start = time.monotonic()
    query = adversarial((QUERY_TOKENS, QUERY_HEADS, HEAD_DIM), 1, 11, 1.5)
    prefix_key = adversarial((PREFIX_KEYS, QUERY_HEADS, HEAD_DIM), 3, 13, 1.25)
    prefix_value = adversarial((PREFIX_KEYS, QUERY_HEADS, HEAD_DIM), 5, 7, 1.0)

    prefix_output, prefix_lse = partial_state(query, prefix_key, prefix_value)
    torch.cuda.synchronize()
    first_gpu_elapsed = time.monotonic() - start

    native_error = None
    try:
        from sgl_kernel.attention import merge_state_v2

        merge_state_v2(prefix_output, prefix_lse, prefix_output, prefix_lse)
    except Exception as error:
        native_error = f"{type(error).__name__}: {error}"

    results = []
    for dtype in (torch.float32, torch.bfloat16):
        dtype_prefix_output = prefix_output.to(dtype)
        for tail_length in TAIL_CASES:
            if tail_length:
                tail_key = adversarial((tail_length, QUERY_HEADS, HEAD_DIM), 7, 5, 1.75)
                tail_value = adversarial(
                    (tail_length, QUERY_HEADS, HEAD_DIM), 9, 3, 1.5
                )
                tail_output, tail_lse = partial_state(query, tail_key, tail_value)
                tail_output = tail_output.to(dtype)
            else:
                tail_output = torch.zeros_like(dtype_prefix_output)
                tail_lse = torch.full_like(prefix_lse, float("-inf"))

            merged_output, merged_lse = merge_state_triton(
                dtype_prefix_output, prefix_lse, tail_output, tail_lse
            )
            torch.cuda.synchronize()

            full_key = (
                torch.cat((prefix_key, tail_key), dim=0) if tail_length else prefix_key
            )
            full_value = (
                torch.cat((prefix_value, tail_value), dim=0)
                if tail_length
                else prefix_value
            )
            full_output, full_lse = partial_state(query, full_key, full_value)
            output_max_abs_diff = (merged_output.float() - full_output).abs().max().item()
            output_mean_abs_diff = (
                merged_output.float() - full_output
            ).abs().mean().item()
            lse_max_abs_diff = (merged_lse - full_lse).abs().max().item()
            lse_mean_abs_diff = (merged_lse - full_lse).abs().mean().item()
            milliseconds = bounded_timing(
                lambda: merge_state_triton(
                    dtype_prefix_output, prefix_lse, tail_output, tail_lse
                )
            )
            results.append(
                {
                    "dtype": str(dtype).removeprefix("torch."),
                    "tail_keys": tail_length,
                    "output_max_abs_diff": output_max_abs_diff,
                    "output_mean_abs_diff": output_mean_abs_diff,
                    "lse_max_abs_diff": lse_max_abs_diff,
                    "lse_mean_abs_diff": lse_mean_abs_diff,
                    "merged_output_all_finite": bool(
                        torch.isfinite(merged_output).all().item()
                    ),
                    "merged_lse_all_finite": bool(
                        torch.isfinite(merged_lse).all().item()
                    ),
                    "mean_milliseconds": milliseconds,
                }
            )

    properties = torch.cuda.get_device_properties(0)
    report = {
        "label": "current mirror checkout real-GPU property control",
        "timestamp_utc": subprocess.check_output(["date", "-u"], text=True).strip(),
        "command": "/opt/venv/bin/python /job/investigate_cascade.py",
        "source_commit": subprocess.check_output(
            ["git", "-C", "/job/sglang", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "rocm_version": getattr(torch.version, "hip", None),
        "gpu": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        },
        "operation": "sglang.kernels.ops.attention.merge_state.merge_state_triton",
        "input_generation": "finite deterministic adversarial sign/magnitude matrix; no NaN/Inf inputs",
        "shape": {
            "query_tokens": QUERY_TOKENS,
            "query_heads": QUERY_HEADS,
            "head_dim": HEAD_DIM,
            "shared_prefix_keys": PREFIX_KEYS,
            "tail_keys": list(TAIL_CASES),
        },
        "reference": "independent float32 einsum softmax and logsumexp over concatenated prefix/tail keys",
        "timing_method": "CUDA events, 3 warmups then mean of 20 calls",
        "numerical_gates": {
            "float32_output_max_abs": 1e-6,
            "float32_lse_max_abs": 1e-5,
            "bfloat16_output_max_abs": 0.02,
            "bfloat16_lse_max_abs": 0.01,
        },
        "first_gpu_execution_elapsed_seconds": first_gpu_elapsed,
        "native_merge_state_v2_error": native_error,
        "results": results,
    }
    with open("/job/cascade-property-results.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
