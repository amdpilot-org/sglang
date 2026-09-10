import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import sglang
import torch
import triton
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd


def cuda_event_ms(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def git_output(args):
    return subprocess.check_output(args, cwd="/job/sglang", text=True).strip()


def module_path(name):
    spec = importlib.util.find_spec(name)
    return spec.origin if spec and spec.origin else None


def run_case(batch_size, seq_len, head_num, head_dim, max_kv_splits, dtype, q, k, v, kv_indptr, kv_indices, reference):
    output = torch.empty_like(q)
    attn_logits = torch.empty(
        (batch_size, head_num, max_kv_splits, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    attn_lse = torch.empty(
        (batch_size, head_num, max_kv_splits),
        dtype=torch.float32,
        device=q.device,
    )
    num_kv_splits = torch.full(
        (batch_size,), max_kv_splits, dtype=torch.int32, device=q.device
    )

    def run():
        decode_attention_fwd(
            q,
            k,
            v,
            output,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale=head_dim**-0.5,
            k_scale=1.0,
            v_scale=1.0,
            logit_cap=0.0,
            page_size=1,
        )

    torch.cuda.synchronize()
    cold_wall_start = time.perf_counter()
    run()
    torch.cuda.synchronize()
    cold_wall_seconds = time.perf_counter() - cold_wall_start
    cold_cuda_event_ms = cuda_event_ms(run, iterations=1)
    warmup_ms = cuda_event_ms(run, iterations=5)
    kernel_ms = cuda_event_ms(run, iterations=20)

    absolute_error = (output.float() - reference).abs()
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "head_num": head_num,
        "head_dim": head_dim,
        "dtype": str(dtype).split(".")[-1],
        "max_kv_splits": max_kv_splits,
        "timing_ms": {
            "cold_wall_seconds": cold_wall_seconds,
            "cold_cuda_event": cold_cuda_event_ms,
            "estimated_compile_overhead_seconds": max(
                0.0, cold_wall_seconds - cold_cuda_event_ms / 1000.0
            ),
            "warmup_5_call_mean": warmup_ms,
            "kernel_20_call_mean": kernel_ms,
        },
        "numerical": {
            "max_abs_error": float(absolute_error.max()),
            "mean_abs_error": float(absolute_error.mean()),
            "allclose_atol_1e-2_rtol_1e-2": bool(
                torch.allclose(output.float(), reference, atol=1e-2, rtol=1e-2)
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    head_num = 16
    head_dim = 128
    batch_sizes = [1, 4]
    seq_lens = [8192, 32768]
    candidates = [1, 2, 4, 8, 16, 32]
    results = []
    benchmark_start = time.perf_counter()

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            total_tokens = batch_size * seq_len
            q = torch.randn(
                batch_size, head_num, head_dim, dtype=dtype, device=device
            )
            k = torch.randn(total_tokens, head_num, head_dim, dtype=dtype, device=device)
            v = torch.randn(total_tokens, head_num, head_dim, dtype=dtype, device=device)
            kv_indices = torch.arange(total_tokens, dtype=torch.int64, device=device)
            kv_indptr = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                seq_len,
                dtype=torch.int32,
                device=device,
            )
            k_reference = (
                k.view(batch_size, seq_len, head_num, head_dim)
                .transpose(1, 2)
                .float()
            )
            v_reference = (
                v.view(batch_size, seq_len, head_num, head_dim)
                .transpose(1, 2)
                .float()
            )
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float().unsqueeze(2),
                k_reference,
                v_reference,
                scale=head_dim**-0.5,
            ).squeeze(2)
            for max_kv_splits in candidates:
                result = run_case(
                    batch_size,
                    seq_len,
                    head_num,
                    head_dim,
                    max_kv_splits,
                    dtype,
                    q,
                    k,
                    v,
                    kv_indptr,
                    kv_indices,
                    reference,
                )
                results.append(result)
                print(json.dumps(result), flush=True)
            del q, k, v, kv_indices, kv_indptr, k_reference, v_reference, reference
            torch.cuda.empty_cache()

    benchmark_elapsed_seconds = time.perf_counter() - benchmark_start
    report = {
        "label": "current-main Triton decode split-cap benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": git_output(["git", "rev-parse", "HEAD"]),
        "source_dirty": bool(git_output(["git", "status", "--porcelain"])),
        "sglang_import_path": sglang.__file__,
        "torch_native_path": torch.__file__,
        "triton_native_path": triton.__file__,
        "sgl_kernel_native_path": module_path("sgl_kernel"),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "architecture": torch.cuda.get_device_capability(0),
            "count": torch.cuda.device_count(),
        },
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "image_identity": {
            "image": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
            "local_image_id": "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
            "source": "operator-provided",
        },
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "shape_matrix": {
            "batch_sizes": batch_sizes,
            "seq_lens": seq_lens,
            "head_num": head_num,
            "head_dim": head_dim,
            "dtype": str(dtype).split(".")[-1],
        },
        "candidates": [f"max_kv_splits={value}" for value in candidates],
        "candidate_count": len(candidates),
        "reference": "float32 scaled_dot_product_attention",
        "reference_tolerance": {"atol": 1e-2, "rtol": 1e-2, "changed": False},
        "timing_method": {
            "compile": "cold wall time minus one-call CUDA event; cache dir is job-private",
            "kernel": "CUDA events around 20 calls after one cold call and 5-call warmup",
            "iterations": 20,
        },
        "architecture_specific_limitations": [
            "Single gfx942 MI300X only; no multi-GPU result.",
            "Direct Triton decode microbenchmark only; no end-to-end serving benchmark.",
            "No full model weights were downloaded.",
        ],
        "benchmark_elapsed_seconds": benchmark_elapsed_seconds,
        "results": results,
        "commands": [
            "PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_decode_attention -q",
            "TRITON_CACHE_DIR=/tmp/sglang-cache-j-d00b06eecff9/triton PYTHONPATH=/job/sglang/python /opt/venv/bin/python reports/j-d00b06eecff9/benchmark_triton_decode.py --output reports/j-d00b06eecff9/benchmark-results.json",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
