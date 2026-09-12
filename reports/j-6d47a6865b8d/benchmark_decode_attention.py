#!/usr/bin/env python3
"""Direct GPU benchmark/correctness check for SGLang Triton decode attention."""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton


def reference(q, k, v, lengths, indices, scale):
    outputs = []
    offset = 0
    groups = q.shape[1] // k.shape[1]
    for batch, length in enumerate(lengths):
        loc = indices[offset : offset + length].long()
        kb = k[loc].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        vb = v[loc].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = torch.einsum("hd,hld->hl", q[batch].float(), kb) * scale
        outputs.append(torch.einsum("hl,hld->hd", scores.softmax(dim=-1), vb))
        offset += length
    return torch.stack(outputs)


def scheduler_splits(lengths, hq, hkv, max_splits):
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    splits = torch.empty(len(lengths), dtype=torch.int32, device="cuda")
    schedule_seq = max(256, triton.next_power_of_2(len(lengths)))
    get_num_kv_splits_triton[(1,)](
        splits,
        seq_lens,
        len(lengths),
        1,
        hq,
        hkv,
        max_splits,
        torch.cuda.get_device_properties(0).multi_processor_count,
        MAX_NUM_SEQ=schedule_seq,
    )
    return splits


def run_case(lengths, hq, hkv, dim, max_splits, seed, iters):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    total = sum(lengths)
    q = torch.randn(len(lengths), hq, dim, dtype=dtype, device="cuda")
    k = torch.randn(total + 17, hkv, dim, dtype=dtype, device="cuda")
    v = torch.randn(total + 17, hkv, dim, dtype=dtype, device="cuda")
    # A non-identity page table exercises gathered KV locations; the final 17 slots
    # remain inaccessible sentinels so masked tail lanes cannot affect the result.
    indices = torch.randperm(total + 17, device="cuda")[:total].to(torch.int64)
    indptr = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    splits = scheduler_splits(lengths, hq, hkv, max_splits)
    out = torch.empty_like(q)
    partial = torch.empty(
        len(lengths), hq, max_splits, dim, dtype=torch.float32, device="cuda"
    )
    lse = torch.empty(
        len(lengths), hq, max_splits, dtype=torch.float32, device="cuda"
    )
    scale = 1.0 / math.sqrt(dim)

    def launch():
        decode_attention_fwd(
            q,
            k,
            v,
            out,
            indptr,
            indices,
            partial,
            lse,
            splits,
            max_splits,
            scale,
            1.0,
            1.0,
        )

    def measure():
        for _ in range(5):
            launch()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(iters):
            start.record()
            launch()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)
        samples.sort()
        return samples

    scheduled_splits = splits.clone()
    samples = measure()
    ref = reference(q, k, v, lengths, indices, scale)
    actual = out.float()
    abs_error = (actual - ref).abs()
    denom = ref.abs().clamp_min(1e-5)
    splits.fill_(1)
    single_split_samples = measure()
    splits.copy_(scheduled_splits)
    return {
        "lengths": lengths,
        "batch": len(lengths),
        "query_heads": hq,
        "kv_heads": hkv,
        "head_dim": dim,
        "scheduler_splits": splits.cpu().tolist(),
        "latency_us_median": samples[len(samples) // 2],
        "latency_us_min": samples[0],
        "single_split_latency_us_median": single_split_samples[
            len(single_split_samples) // 2
        ],
        "single_split_latency_us_min": single_split_samples[0],
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
        "max_relative_error": (abs_error / denom).max().item(),
        "allclose_atol_2e-2_rtol_2e-2": torch.allclose(
            actual, ref, atol=2e-2, rtol=2e-2
        ),
        "finite": bool(torch.isfinite(actual).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=40)
    args = parser.parse_args()
    cases = []
    # The issue's Llama-3.1-8B GQA decode shape, extended to genuinely long context.
    for length in (200, 2000, 8192, 32768):
        cases.append(([length], 32, 8, 128))
    # Batch pressure plus irregular boundaries around the stage-1 block sizes.
    cases.extend(
        [
            ([200] * 8, 32, 8, 128),
            ([2000] * 8, 32, 8, 128),
            ([31, 32, 33, 63, 64, 65, 199, 200], 32, 8, 128),
            ([1999, 2000, 2001], 32, 8, 128),
            # MHA takes the separate non-grouped stage-1 implementation.
            ([200], 32, 32, 128),
            ([2000], 32, 32, 128),
        ]
    )
    started = time.time()
    results = [
        run_case(lengths, hq, hkv, dim, 32, 1234 + i, args.iters)
        for i, (lengths, hq, hkv, dim) in enumerate(cases)
    ]
    payload = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "device": torch.cuda.get_device_name(0),
            "device_properties": str(torch.cuda.get_device_properties(0)),
        },
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
