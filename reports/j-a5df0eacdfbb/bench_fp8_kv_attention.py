#!/usr/bin/env python3

import inspect
import json
import math
import os
import statistics
import sys
import time

import torch
import triton
from aiter.ops.triton.attention.unified_attention import (
    select_3d_config,
    unified_attention,
    use_2d_kernel,
)
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.types import e4m3_dtype
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, scaled_fp8_quant


SEQ_LENS = [1024, 2048, 4096, 8192]
BATCH = 4
Q_HEADS = 16
KV_HEADS = 1
HEAD_DIM = 256
PAGE_SIZE = 16
WARMUP_CALLS = 5
MEASURED_CALLS = 20
SEED = 31783


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def _dispatch_config(seq_len, dtype):
    cu_count = get_num_sms()
    target_num_prgms = cu_count * 4
    total_num_q_blocks = BATCH
    num_2d_prgms = total_num_q_blocks * KV_HEADS
    use_2d = use_2d_kernel(
        HEAD_DIM,
        0,
        True,
        1,
        seq_len,
        target_num_prgms,
        num_2d_prgms,
    )
    attn_config, reduce_config = select_3d_config(
        HEAD_DIM,
        PAGE_SIZE,
        seq_len,
        target_num_prgms,
        num_2d_prgms,
        dtype,
        dtype,
        False,
        1,
        0,
    )
    num_segments = attn_config["NUM_SEGMENTS_PER_SEQ"]
    if num_segments > 1:
        segment_output_bytes = (
            BATCH
            * Q_HEADS
            * num_segments
            * triton.next_power_of_2(HEAD_DIM)
            * 4
        )
        segment_max_bytes = BATCH * Q_HEADS * num_segments * 4
        segment_expsum_bytes = BATCH * Q_HEADS * num_segments * 4
    else:
        segment_output_bytes = 0
        segment_max_bytes = 0
        segment_expsum_bytes = 0
    num_queries_per_kv = Q_HEADS // KV_HEADS
    block_m = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    block_q = block_m // num_queries_per_kv
    return {
        "kernel": (
            "kernel_unified_attention_2d"
            if use_2d
            else "kernel_unified_attention_3d"
        ),
        "use_2d": use_2d,
        "block_m": block_m,
        "block_q": block_q,
        "attn_config": attn_config,
        "reduce_config": reduce_config,
        "num_segments": num_segments,
        "tile_size": attn_config["TILE_SIZE"],
        "num_warps": attn_config["num_warps"],
        "num_stages": attn_config["num_stages"],
        "waves_per_eu": attn_config["waves_per_eu"],
        "kernel_workspace_bytes": (
            segment_output_bytes + segment_max_bytes + segment_expsum_bytes
        ),
    }


def _make_case(seq_len, seed):
    torch.manual_seed(seed)
    device = "cuda"
    base = torch.randn(BATCH, HEAD_DIM, device=device, dtype=torch.float32)
    q = (
        base[:, None, :].expand(-1, Q_HEADS, -1)
        + 0.01
        * torch.randn(
            BATCH,
            Q_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    k = 0.1 * torch.randn(
        BATCH,
        seq_len,
        KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    k[:, 0, 0, :] = 2 * base
    k = k.to(torch.bfloat16)
    v = (
        0.75
        + 0.25
        * torch.randn(
            BATCH,
            seq_len,
            KV_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)

    fp8_max = torch.finfo(fp8_dtype).max
    q_scale = (q.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)
    k_scale = (k.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)
    v_scale = (v.abs().float().amax() / fp8_max).clamp(min=1e-9).view(1)

    q_fp8, _ = scaled_fp8_quant(q.reshape(BATCH, -1), q_scale)
    k_fp8, _ = scaled_fp8_quant(k.reshape(-1, HEAD_DIM), k_scale)
    v_fp8, _ = scaled_fp8_quant(v.reshape(-1, HEAD_DIM), v_scale)
    q_fp8 = q_fp8.view(BATCH, Q_HEADS, HEAD_DIM)
    k_fp8 = k_fp8.view(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM)
    v_fp8 = v_fp8.view(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM)

    pages_per_seq = seq_len // PAGE_SIZE
    block_table = torch.arange(
        BATCH * pages_per_seq, dtype=torch.int32, device=device
    ).view(BATCH, pages_per_seq)
    cu_seqlens_q = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
    seqused_k = torch.full(
        (BATCH,), seq_len, dtype=torch.int32, device=device
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "q_fp8": q_fp8,
        "k_fp8": k_fp8,
        "v_fp8": v_fp8,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "block_table": block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
    }


def _reference_bf16(q, k, v):
    scores = torch.einsum(
        "bhd,btd->bht",
        q.float(),
        k[:, :, 0, :].float(),
    ) / math.sqrt(HEAD_DIM)
    return torch.einsum(
        "bht,btd->bhd",
        torch.softmax(scores, dim=-1),
        v[:, :, 0, :].float(),
    )


def _reference_fp8(case, seq_len):
    q = case["q_fp8"].float() * case["q_scale"]
    k = case["k_fp8"][case["block_table"]].float() * case["k_scale"]
    v = case["v_fp8"][case["block_table"]].float() * case["v_scale"]
    k = k.reshape(BATCH, seq_len, KV_HEADS, HEAD_DIM)
    v = v.reshape(BATCH, seq_len, KV_HEADS, HEAD_DIM)
    scores = torch.einsum(
        "bhd,btd->bht",
        q,
        k[:, :, 0, :],
    ) / math.sqrt(HEAD_DIM)
    return torch.einsum(
        "bht,btd->bhd",
        torch.softmax(scores, dim=-1),
        v[:, :, 0, :],
    )


def _accuracy(actual, expected, dtype):
    actual = actual.float()
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    mismatch = diff > (0.15 + 0.15 * expected.abs())
    mismatch_fraction = mismatch.float().mean().item()
    cosine = (
        torch.nn.functional.cosine_similarity(
            actual.flatten(),
            expected.flatten(),
            dim=0,
        ).item()
    )
    finite = bool(torch.isfinite(actual).all())
    if dtype == torch.bfloat16:
        passed = finite and max_abs <= 0.02 and cosine >= 0.9999
    else:
        passed = finite and mismatch_fraction <= 0.005 and cosine >= 0.99
    return {
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "mismatch_fraction": mismatch_fraction,
        "cosine": cosine,
        "passed": passed,
        "rejected": not passed,
    }


def _time_kernel(callable_fn):
    for _ in range(WARMUP_CALLS):
        callable_fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(MEASURED_CALLS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callable_fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    ci95 = 1.96 * stdev / math.sqrt(len(samples))
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": mean,
        "stdev_ms": stdev,
        "ci95_mean_ms": ci95,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _run_unified(case, seq_len, dtype):
    if dtype == torch.bfloat16:
        q = case["q"]
        k = case["k"].reshape(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM)
        v = case["v"].reshape(-1, PAGE_SIZE, KV_HEADS, HEAD_DIM)
        q_descale = None
        k_descale = None
        v_descale = None
    else:
        q = case["q_fp8"]
        k = case["k_fp8"]
        v = case["v_fp8"]
        q_descale = case["q_scale"]
        k_descale = case["k_scale"]
        v_descale = case["v_scale"]
    out = torch.empty(
        BATCH,
        Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    def call():
        unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=case["cu_seqlens_q"],
            max_seqlen_q=1,
            seqused_k=case["seqused_k"],
            max_seqlen_k=seq_len,
            softmax_scale=1 / math.sqrt(HEAD_DIM),
            causal=True,
            window_size=(-1, -1),
            block_table=case["block_table"],
            softcap=0,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    return out, call



def _run_case(seq_len, seed):
    case = _make_case(seq_len, seed)
    results = {}
    for dtype_name, dtype in (
        ("bf16", torch.bfloat16),
        ("fp8", fp8_dtype),
    ):
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        out, call = _run_unified(case, seq_len, dtype)
        call()
        torch.cuda.synchronize()
        dispatch = _dispatch_config(seq_len, dtype)
        if dtype == torch.bfloat16:
            expected = _reference_bf16(case["q"], case["k"], case["v"])
        else:
            expected = _reference_fp8(case, seq_len)
        accuracy = _accuracy(out, expected, dtype)
        timing = _time_kernel(call)
        allocated_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        peak_allocated = torch.cuda.max_memory_allocated()
        cache_bytes = (
            _tensor_bytes(case["q"])
            + _tensor_bytes(case["k"])
            + _tensor_bytes(case["v"])
            + _tensor_bytes(case["q_fp8"])
            + _tensor_bytes(case["k_fp8"])
            + _tensor_bytes(case["v_fp8"])
            + _tensor_bytes(case["block_table"])
            + _tensor_bytes(out)
        )
        results[dtype_name] = {
            "dtype": str(dtype),
            "dispatch": dispatch,
            "accuracy": accuracy,
            "timing": timing,
            "memory": {
                "cache_bytes": cache_bytes,
                "kernel_workspace_bytes": dispatch["kernel_workspace_bytes"],
                "allocated_before_bytes": allocated_before,
                "allocated_after_bytes": allocated_after,
                "reserved_before_bytes": reserved_before,
                "reserved_after_bytes": reserved_after,
                "peak_allocated_bytes": peak_allocated,
            },
        }
    return {
        "case_id": f"seq{seq_len}",
        "seq_len": seq_len,
        "seed": seed,
        "batch": BATCH,
        "num_query_heads": Q_HEADS,
        "num_kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "runs": results,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required")
    start_wall = time.monotonic()
    cases = [_run_case(seq_len, SEED + index) for index, seq_len in enumerate(SEQ_LENS)]
    elapsed_wall = time.monotonic() - start_wall
    device = torch.cuda.get_device_properties(0)
    report = {
        "label": "bounded gfx942 BF16/FP8 KV reduced-attention comparison",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": tuple(torch.cuda.get_device_capability(0)),
            "gpu_total_bytes": device.total_memory,
            "multi_processor_count": device.multi_processor_count,
            "sglang_source": "/job/sglang",
            "aiter_source": inspect.getsourcefile(unified_attention),
            "image": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        },
        "method": {
            "kernel": "aiter unified_attention",
            "reference": (
                "independent float32 einsum/softmax; FP8 reference dequantizes "
                "q/k/v with explicit scales"
            ),
            "input_data": (
                "identical generated BF16 base tensors per sequence-length case; "
                "FP8 tensors are quantized from the same base"
            ),
            "timing": (
                "CUDA events around real forward; 5 warmups and 20 measured calls; "
                "milliseconds"
            ),
            "shared_hardware_uncertainty": (
                "all cases ran on the same MI300X; report sample stdev and 95% "
                "normal CI for the mean"
            ),
        },
        "numerical_gates": {
            "bf16": "finite, max_abs <= 0.02, cosine >= 0.9999",
            "fp8": "finite, mismatch_fraction <= 0.005, cosine >= 0.99",
        },
        "limits": {
            "workload_cases": len(SEQ_LENS),
            "kernel_dispatch_configurations": len(SEQ_LENS),
            "warmup_calls_per_run": WARMUP_CALLS,
            "measured_calls_per_run": MEASURED_CALLS,
            "weights_bytes": 0,
            "total_live_allocations_limit_bytes": 48 * 1024**3,
            "wall_limit_seconds": 7200,
        },
        "cases": cases,
        "wall_elapsed_seconds": elapsed_wall,
    }
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results.json",
    )
    with open(output_path, "w") as output_file:
        json.dump(report, output_file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
