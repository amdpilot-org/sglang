import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import sglang
import torch
import triton
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd_grouped
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton


def load_pr_benchmark():
    path = Path("/job/sglang/benchmark/kernels/attention/bench_triton_decode_splitk.py")
    spec = importlib.util.spec_from_file_location("pr35801_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def run_mode(mode, cap, q, k, v, kv_indptr, kv_indices, reference, pr_benchmark):
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    seq_lens = torch.full((batch_size,), kv_indices.numel() // batch_size, dtype=torch.int32, device=q.device)
    num_kv_splits = torch.empty(batch_size, dtype=torch.int32, device=q.device)
    device_core_count = torch.cuda.get_device_properties(0).multi_processor_count
    kernel = (
        pr_benchmark.get_num_kv_splits_triton_legacy
        if mode == "legacy"
        else get_num_kv_splits_triton
    )
    kernel[(1,)](
        num_kv_splits,
        seq_lens,
        batch_size,
        1,
        num_heads,
        num_kv_heads,
        cap,
        device_core_count,
        MAX_NUM_SEQ=32,
    )
    actual_splits = int(num_kv_splits[0].item())
    output = torch.empty_like(q)
    attn_logits = torch.empty(
        (batch_size, num_heads, cap, head_dim), dtype=torch.float32, device=q.device
    )
    attn_lse = torch.empty(
        (batch_size, num_heads, cap), dtype=torch.float32, device=q.device
    )

    def run():
        decode_attention_fwd_grouped(
            q,
            k,
            v,
            output,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            cap,
            head_dim**-0.5,
            1.0,
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
        "mode": mode,
        "max_kv_splits_cap": cap,
        "actual_splits": actual_splits,
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
            "allclose_atol_3e-2_rtol_3e-2": bool(
                torch.allclose(output.float(), reference, atol=3e-2, rtol=3e-2)
            ),
        },
    }


def main():
    parser_output = Path("/job/sglang/reports/j-d00b06eecff9/pr35801-results.json")
    pr_benchmark = load_pr_benchmark()
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16
    batch_size = 1
    seq_len = 32768
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    total_tokens = batch_size * seq_len
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    kv_indices = torch.arange(total_tokens, dtype=torch.int64, device=device)
    kv_indptr = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)
    k_reference = (
        k.view(batch_size, seq_len, num_kv_heads, head_dim)
        .transpose(1, 2)
        .float()
    )
    v_reference = (
        v.view(batch_size, seq_len, num_kv_heads, head_dim)
        .transpose(1, 2)
        .float()
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.float().unsqueeze(2),
        k_reference,
        v_reference,
        scale=head_dim**-0.5,
        enable_gqa=True,
    ).squeeze(2)

    results = []
    benchmark_start = time.perf_counter()
    for cap in [8, 32]:
        for mode in ["legacy", "dynamic"]:
            result = run_mode(
                mode,
                cap,
                q,
                k,
                v,
                kv_indptr,
                kv_indices,
                reference,
                pr_benchmark,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    benchmark_elapsed_seconds = time.perf_counter() - benchmark_start
    report = {
        "label": "PR 35801 dynamic split-K benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pr": 35801,
        "pr_url": "https://github.com/sgl-project/sglang/pull/35801",
        "tested_commit": git_output(["git", "rev-parse", "HEAD"]),
        "tested_commit_parent": git_output(["git", "rev-parse", "HEAD^"]),
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
        "shape": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": str(dtype).split(".")[-1],
        },
        "reference": "float32 scaled_dot_product_attention with enable_gqa=True",
        "reference_tolerance": {"atol": 3e-2, "rtol": 3e-2, "changed": False},
        "timing_method": {
            "compile": "cold wall time minus one-call CUDA event; cache dir is job-private",
            "kernel": "CUDA events around 20 calls after one cold call and 5-call warmup",
            "iterations": 20,
        },
        "architecture_specific_limitations": [
            "Single gfx942 MI300X only; no multi-GPU result.",
            "Direct grouped Triton decode microbenchmark only; no end-to-end serving benchmark.",
            "No full model weights were downloaded.",
        ],
        "benchmark_elapsed_seconds": benchmark_elapsed_seconds,
        "results": results,
        "commands": [
            "TRITON_CACHE_DIR=/tmp/sglang-cache-j-d00b06eecff9/triton-pr35801 PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest test/registered/attention/test_dynamic_split_triton_decode.py -q",
            "TRITON_CACHE_DIR=/tmp/sglang-cache-j-d00b06eecff9/triton-pr35801-run2 PYTHONPATH=/job/sglang/python /opt/venv/bin/python reports/j-d00b06eecff9/benchmark_pr35801.py",
        ],
    }
    parser_output.parent.mkdir(parents=True, exist_ok=True)
    parser_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
