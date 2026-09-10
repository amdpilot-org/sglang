import argparse
import inspect
import json
import math
import time
from pathlib import Path

import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)


def _tile(values, shape, device, dtype):
    count = math.prod(shape)
    flat = torch.tensor(values, dtype=torch.float32, device=device).repeat(
        (count + len(values) - 1) // len(values)
    )[:count]
    return flat.reshape(shape).to(dtype)


def _reference(q, k, v, indices, scale):
    outputs = []
    for row in range(q.shape[0]):
        valid = indices[row] >= 0
        slots = indices[row, valid].long()
        keys = k.index_select(0, slots).float()
        values = v.index_select(0, slots).float()
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probabilities, values))
    return torch.stack(outputs)


def _metrics(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def _timed_call(function, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = function(**kwargs)
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000.0


def _reference_close(actual, expected):
    return bool(
        torch.allclose(
            actual.float(), expected.float(), rtol=3e-2, atol=3e-2
        )
        and torch.isfinite(actual).all().item()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("base", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(9079)
    q = _tile(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        (3, 4, 128),
        device,
        torch.bfloat16,
    )
    k_fp8 = _tile(
        [-448.0, -1.0, -2.0**-9, 0.0, 2.0**-9, 1.0, 448.0],
        (6, 1, 128),
        device,
        torch.float8_e4m3fn,
    )
    v_fp8 = _tile(
        [-448.0, -1.0, -2.0**-9, 0.0, 2.0**-9, 1.0, 448.0],
        (6, 1, 128),
        device,
        torch.float8_e4m3fn,
    )
    k_scale = 0.25
    v_scale = 0.5
    chunk_indices = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 3, 4], [1, 3, 4, 5]],
        dtype=torch.int32,
        device=device,
    )
    cu_q = torch.tensor([0, 3], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 6], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([6], dtype=torch.int32, device=device)
    softmax_scale = 128**-0.5

    signature = inspect.signature(sparse_gqa_fwd_interface_triton_ck)
    supports_scales = "k_scale" in signature.parameters
    prefill_k_fp8 = k_fp8[:3].contiguous()
    prefill_v_fp8 = v_fp8[:3].contiguous()
    prefill_indices = torch.tensor(
        [[0, -1, -1], [0, 1, -1], [0, 1, 2]],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32, device=device)
    chunk_common = {
        "q": q,
        "indices": chunk_indices,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "kv_lens": kv_lens,
        "scale": softmax_scale,
    }

    if args.mode == "base":
        try:
            sparse_gqa_fwd_interface_triton_ck(
                q, k_fp8, v_fp8, chunk_indices, cu_q, cu_k, kv_lens, softmax_scale
            )
        except Exception as error:
            result = {
                "mode": "base",
                "supports_scales": supports_scales,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return
        raise RuntimeError("base mixed-dtype call unexpectedly succeeded")

    if not supports_scales:
        raise RuntimeError("candidate mode requires k_scale/v_scale support")

    prefill_k_bf16 = (prefill_k_fp8.float() * k_scale).to(torch.bfloat16)
    prefill_v_bf16 = (prefill_v_fp8.float() * v_scale).to(torch.bfloat16)
    k_bf16 = (k_fp8.float() * k_scale).to(torch.bfloat16)
    v_bf16 = (v_fp8.float() * v_scale).to(torch.bfloat16)
    sparse_gqa_fwd_interface_triton(
        q=q,
        k=prefill_k_fp8,
        v=prefill_v_fp8,
        max_seqlen_k=3,
        indices=prefill_indices,
        cu_seqlens=cu_seqlens,
        scale=softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    sparse_gqa_fwd_interface_triton(
        q=q,
        k=prefill_k_bf16,
        v=prefill_v_bf16,
        max_seqlen_k=3,
        indices=prefill_indices,
        cu_seqlens=cu_seqlens,
        scale=softmax_scale,
    )
    sparse_gqa_fwd_interface_triton_ck(
        k=k_fp8,
        v=v_fp8,
        k_scale=k_scale,
        v_scale=v_scale,
        **chunk_common,
    )
    sparse_gqa_fwd_interface_triton_ck(
        k=k_bf16,
        v=v_bf16,
        **chunk_common,
    )
    fp8_prefill, fp8_prefill_ms = _timed_call(
        sparse_gqa_fwd_interface_triton,
        q=q,
        k=prefill_k_fp8,
        v=prefill_v_fp8,
        max_seqlen_k=3,
        indices=prefill_indices,
        cu_seqlens=cu_seqlens,
        scale=softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    fp8_chunk, fp8_chunk_ms = _timed_call(
        sparse_gqa_fwd_interface_triton_ck,
        k=k_fp8,
        v=v_fp8,
        k_scale=k_scale,
        v_scale=v_scale,
        **chunk_common,
    )
    bf16_prefill, bf16_prefill_ms = _timed_call(
        sparse_gqa_fwd_interface_triton,
        q=q,
        k=prefill_k_bf16,
        v=prefill_v_bf16,
        max_seqlen_k=3,
        indices=prefill_indices,
        cu_seqlens=cu_seqlens,
        scale=softmax_scale,
    )
    bf16_chunk, bf16_chunk_ms = _timed_call(
        sparse_gqa_fwd_interface_triton_ck,
        k=k_bf16,
        v=v_bf16,
        **chunk_common,
    )
    prefill_reference = _reference(
        q,
        prefill_k_bf16.float(),
        prefill_v_bf16.float(),
        prefill_indices,
        softmax_scale,
    )
    chunk_reference = _reference(
        q,
        k_bf16.float(),
        v_bf16.float(),
        chunk_indices,
        softmax_scale,
    )
    result = {
        "mode": "candidate",
        "supports_scales": supports_scales,
        "input_contract": {
            "q_shape": list(q.shape),
            "kv_shape": list(k_fp8.shape),
            "q_dtype": str(q.dtype),
            "kv_dtype": str(k_fp8.dtype),
            "k_scale": k_scale,
            "v_scale": v_scale,
            "softmax_scale": softmax_scale,
            "finite_values": "deterministic 7-value sign/subnormal/max matrix",
        },
        "prefill": {
            "fp8_vs_bf16": _metrics(fp8_prefill, bf16_prefill),
            "fp8_vs_reference": _metrics(fp8_prefill, prefill_reference),
            "bf16_vs_reference": _metrics(bf16_prefill, prefill_reference),
        },
        "chunk_prefill": {
            "fp8_vs_bf16": _metrics(fp8_chunk, bf16_chunk),
            "fp8_vs_reference": _metrics(fp8_chunk, chunk_reference),
            "bf16_vs_reference": _metrics(bf16_chunk, chunk_reference),
        },
        "timing_ms": {
            "fp8_prefill": fp8_prefill_ms,
            "bf16_prefill": bf16_prefill_ms,
            "fp8_chunk": fp8_chunk_ms,
            "bf16_chunk": bf16_chunk_ms,
        },
        "gate": {
            "fp8_vs_bf16_exact": True,
            "reference_rtol": 3e-2,
            "reference_atol": 3e-2,
            "passed": all([
                torch.equal(fp8_prefill, bf16_prefill),
                torch.equal(fp8_chunk, bf16_chunk),
                _reference_close(fp8_prefill, prefill_reference),
                _reference_close(bf16_prefill, prefill_reference),
                _reference_close(fp8_chunk, chunk_reference),
                _reference_close(bf16_chunk, chunk_reference),
                torch.isfinite(fp8_prefill).all().item(),
                torch.isfinite(bf16_prefill).all().item(),
                torch.isfinite(fp8_chunk).all().item(),
                torch.isfinite(bf16_chunk).all().item(),
            ]),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
