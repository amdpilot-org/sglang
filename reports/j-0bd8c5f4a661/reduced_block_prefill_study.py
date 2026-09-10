#!/usr/bin/env python3
"""Bounded gfx942 BF16/FP8-KV prefill study for a dummy-model direct path.

This benchmark intentionally makes no model-quality claim.  It measures the
installed AITER linear prefill kernel with explicit, locally generated Q/K/V
values and compares each output against an independent dequantized SDPA
reference.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.profiler import ProfilerActivity, profile


CASES = [
    {"name": "b8_s4096_h32_kv8_d128", "batch": 8, "seq": 4096, "q_heads": 32, "kv_heads": 8, "head_dim": 128},
    {"name": "b16_s2048_h32_kv8_d128", "batch": 16, "seq": 2048, "q_heads": 32, "kv_heads": 8, "head_dim": 128},
    {"name": "b32_s1024_h32_kv8_d128", "batch": 32, "seq": 1024, "q_heads": 32, "kv_heads": 8, "head_dim": 128},
    {"name": "b8_s4096_h64_kv16_d128", "batch": 8, "seq": 4096, "q_heads": 64, "kv_heads": 16, "head_dim": 128},
    {"name": "b16_s2048_h64_kv16_d128", "batch": 16, "seq": 2048, "q_heads": 64, "kv_heads": 16, "head_dim": 128},
    {"name": "b32_s1024_h64_kv16_d128", "batch": 32, "seq": 1024, "q_heads": 64, "kv_heads": 16, "head_dim": 128},
]

WARM_RUNS = 5
SEED = 31783
FP8_SCALE = 0.02
MAX_ABS_ERROR_GATE = 0.1
MEAN_ABS_ERROR_GATE = 0.01
LOCAL_GENERATED_LIMIT_BYTES = 4 * 1024**3
TOTAL_LIVE_ALLOCATION_LIMIT_BYTES = 48 * 1024**3


def tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def git_head(repository: str) -> str:
    return subprocess.check_output(
        ["git", "-C", repository, "rev-parse", "HEAD"], text=True
    ).strip()


def independent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch: int,
    seq: int,
    head_dim: int,
    mode: str,
    fp8_scale: float,
) -> torch.Tensor:
    references = []
    for request_index in range(batch):
        start = request_index * seq
        stop = start + seq
        if mode == "bf16":
            q_ref = q[start:stop]
            k_ref = k[start:stop]
            v_ref = v[start:stop]
        else:
            q_ref = q[start:stop].float() * fp8_scale
            k_ref = k[start:stop].float() * fp8_scale
            v_ref = v[start:stop].float() * fp8_scale
        reference = scaled_dot_product_attention(
            q_ref.transpose(0, 1).unsqueeze(0),
            k_ref.transpose(0, 1).unsqueeze(0),
            v_ref.transpose(0, 1).unsqueeze(0),
            is_causal=True,
            scale=head_dim**-0.5,
            enable_gqa=True,
        ).squeeze(0).transpose(0, 1)
        references.append(reference.to(torch.bfloat16))
    return torch.cat(references, dim=0)


def profile_dispatch(run_forward) -> list[dict]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        run_forward()
        torch.cuda.synchronize()
    events = [
        event
        for event in profiler.key_averages()
        if event.self_device_time_total > 0
        and any(token in event.key.lower() for token in ("mha", "prefill", "attention"))
    ]
    events.sort(key=lambda event: event.self_device_time_total, reverse=True)
    return [
        {
            "name": event.key,
            "self_device_time_us": event.self_device_time_total,
            "calls": event.count,
        }
        for event in events[:5]
    ]


def run_mode(case: dict, mode: str, fp8_dtype: torch.dtype, first_mode_call: bool) -> dict:
    torch.manual_seed(SEED)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    batch = case["batch"]
    seq = case["seq"]
    q_heads = case["q_heads"]
    kv_heads = case["kv_heads"]
    head_dim = case["head_dim"]
    total_tokens = batch * seq
    device = "cuda"

    q_bf16 = torch.randn(total_tokens, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(total_tokens, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(total_tokens, kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    if mode == "bf16":
        q, k, v = q_bf16, k_bf16, v_bf16
        q_descale = k_descale = v_descale = None
    else:
        fp8_max = torch.finfo(fp8_dtype).max
        q = torch.clamp(q_bf16.float() / FP8_SCALE, -fp8_max, fp8_max).to(fp8_dtype)
        k = torch.clamp(k_bf16.float() / FP8_SCALE, -fp8_max, fp8_max).to(fp8_dtype)
        v = torch.clamp(v_bf16.float() / FP8_SCALE, -fp8_max, fp8_max).to(fp8_dtype)
        q_descale = torch.tensor([FP8_SCALE], dtype=torch.float32, device=device)
        k_descale = q_descale
        v_descale = q_descale

    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=device) * seq
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)

    from aiter.ops.mha import mha_batch_prefill_func

    def run_forward():
        return mha_batch_prefill_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            kv_indices,
            seq,
            seq,
            causal=True,
            softmax_scale=head_dim**-0.5,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    output = run_forward()
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - cold_start

    warm_seconds = []
    for _ in range(WARM_RUNS):
        start = time.perf_counter()
        output = run_forward()
        torch.cuda.synchronize()
        warm_seconds.append(time.perf_counter() - start)

    dispatch = profile_dispatch(run_forward)
    reference = independent_reference(
        q, k, v, batch, seq, head_dim, mode, FP8_SCALE
    )
    difference = (output.float() - reference.float()).abs()
    max_abs_error = difference.max().item()
    mean_abs_error = difference.mean().item()
    gate_passed = (
        max_abs_error <= MAX_ABS_ERROR_GATE
        and mean_abs_error <= MEAN_ABS_ERROR_GATE
    )

    warm_median = statistics.median(warm_seconds)
    approximate_flops = 2.0 * batch * seq * seq * q_heads * head_dim
    generated_bytes = tensor_bytes(q, k, v, output)
    cache_bytes = tensor_bytes(k, v)
    peak_allocated = torch.cuda.max_memory_allocated()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    if generated_bytes > LOCAL_GENERATED_LIMIT_BYTES:
        raise RuntimeError(f"generated tensors exceed 4 GiB: {generated_bytes}")
    if peak_allocated > TOTAL_LIVE_ALLOCATION_LIMIT_BYTES:
        raise RuntimeError(f"live allocation exceeds 48 GiB: {peak_allocated}")

    result = {
        "case": case["name"],
        "mode": mode,
        "dimensions": {
            "batch_size": batch,
            "sequence_length": seq,
            "total_query_tokens": total_tokens,
            "query_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        "dtypes": {
            "query": str(q.dtype),
            "key_cache": str(k.dtype),
            "value_cache": str(v.dtype),
            "output": str(output.dtype),
            "reference_output": str(reference.dtype),
        },
        "explicit_values_and_scales": {
            "seed": SEED,
            "distribution": "torch.randn",
            "fp8_scale": FP8_SCALE if mode == "fp8" else 1.0,
            "q_descale": FP8_SCALE if mode == "fp8" else 1.0,
            "k_descale": FP8_SCALE if mode == "fp8" else 1.0,
            "v_descale": FP8_SCALE if mode == "fp8" else 1.0,
        },
        "accuracy": {
            "gate": f"max_abs_error <= {MAX_ABS_ERROR_GATE} and mean_abs_error <= {MEAN_ABS_ERROR_GATE}",
            "passed": gate_passed,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "claim": "numerical comparison only; no model-quality claim",
        },
        "timing": {
            "method": "time.perf_counter around synchronized real AITER prefill calls",
            "cold_seconds": cold_seconds,
            "cold_includes_jit_compilation": first_mode_call,
            "warm_runs": WARM_RUNS,
            "warm_seconds": warm_seconds,
            "warm_median_seconds": warm_median,
            "warm_mean_seconds": statistics.mean(warm_seconds),
            "query_tokens_per_second": total_tokens / warm_median,
            "approximate_causal_attention_flops_per_forward": approximate_flops,
            "approximate_attention_tflops_per_second": approximate_flops / warm_median / 1e12,
        },
        "allocation": {
            "query_bytes": tensor_bytes(q),
            "key_cache_bytes": tensor_bytes(k),
            "value_cache_bytes": tensor_bytes(v),
            "output_bytes": tensor_bytes(output),
            "reference_output_bytes": tensor_bytes(reference),
            "generated_tensor_bytes": generated_bytes,
            "torch_cuda_allocated_bytes": allocated,
            "torch_cuda_reserved_bytes": reserved,
            "torch_cuda_peak_allocated_bytes": peak_allocated,
        },
        "dispatch": {
            "function": "aiter.ops.mha.mha_batch_prefill_func",
            "layout": "linear 3D NHD, page_size=1",
            "causal": True,
            "profiled_device_events": dispatch,
        },
    }

    del output, reference, difference, q, k, v, q_bf16, k_bf16, v_bf16
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    from aiter.ops.mha import mha_batch_prefill_func
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

    properties = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    results = []
    first_mode_seen = {"bf16": True, "fp8": True}

    for case in CASES:
        for mode in ("bf16", "fp8"):
            print(f"running {case['name']} {mode}", flush=True)
            result = run_mode(case, mode, fp8_dtype, first_mode_seen[mode])
            first_mode_seen[mode] = False
            results.append(result)

    report = {
        "schema_version": 1,
        "label": "gfx942 dummy-model direct AITER linear prefill BF16/FP8-KV study",
        "campaign": "repo-e2e-20260909",
        "coordination_tracker": "amdpilot-org/amdpilotv2 issue 402",
        "upstream_context": {
            "issue": "sgl-project/sglang issue 31783",
            "related_pr_checked": "sgl-project/sglang PR 32576",
            "related_pr_scope": "resolution-time DSA compatibility table; no GPU kernel change",
            "duplicate_fix": False,
        },
        "source": {
            "repository": str(args.repository.resolve()),
            "commit": git_head(str(args.repository.resolve())),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "sglang_import": __import__("sglang").__file__,
            "torch_import": torch.__file__,
            "aiter_import": __import__("aiter").__file__,
            "aiter_mha_import": __import__("aiter.ops.mha", fromlist=["mha_batch_prefill_func"]).__file__,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "gcn_arch_name": properties.gcnArchName,
            "capability": list(torch.cuda.get_device_capability(0)),
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "free_memory_bytes_at_start": free_bytes,
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "image_identity": {
            "required_image": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "required_local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
            "verification": "operator-provided local image ID; not treated as a pullable registry digest",
        },
        "command": {
            "argv": sys.argv,
            "cache_environment": {
                "AITER_JIT_DIR": os.environ.get("AITER_JIT_DIR"),
                "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
                "FLYDSL_RUNTIME_CACHE_DIR": os.environ.get("FLYDSL_RUNTIME_CACHE_DIR"),
            },
        },
        "limits": {
            "wall_limit_seconds": 7200,
            "workload_cases": len(CASES),
            "modes_per_case": 2,
            "warm_real_forwards_per_case_mode": WARM_RUNS,
            "profiled_real_forwards_per_case_mode": 1,
            "local_generated_tensor_limit_bytes": LOCAL_GENERATED_LIMIT_BYTES,
            "total_live_allocation_limit_bytes": TOTAL_LIVE_ALLOCATION_LIMIT_BYTES,
            "model_weights_downloaded": False,
            "synthetic_burn_or_unbounded_loop": False,
        },
        "numerical_gates": {
            "bf16_and_fp8": f"max_abs_error <= {MAX_ABS_ERROR_GATE} and mean_abs_error <= {MEAN_ABS_ERROR_GATE}",
            "interpretation": "per-case output comparison only; not a model-quality gate",
        },
        "raw_results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
