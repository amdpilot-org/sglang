"""Bounded MI300X benchmark for shared-prefix/tail attention decomposition."""

import json
import sys

import torch


def _timed_call(call, warmup=3, iterations=20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    total_ms = 0.0
    for _ in range(iterations):
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / iterations


def main():
    torch.manual_seed(1715)
    device = torch.device("cuda")
    batch_size, prefix_len, tail_len, num_heads, head_dim = 8, 512, 64, 8, 64
    dtype = torch.float16
    scale = head_dim**-0.5

    query = torch.randn(
        batch_size, num_heads, head_dim, device=device, dtype=dtype
    )
    prefix_key = torch.randn(
        prefix_len, num_heads, head_dim, device=device, dtype=dtype
    )
    prefix_value = torch.randn(
        prefix_len, num_heads, head_dim, device=device, dtype=dtype
    )
    tail_key = torch.randn(
        batch_size, tail_len, num_heads, head_dim, device=device, dtype=dtype
    )
    tail_value = torch.randn(
        batch_size, tail_len, num_heads, head_dim, device=device, dtype=dtype
    )
    full_value = torch.cat(
        [prefix_value.unsqueeze(0).expand(batch_size, -1, -1, -1), tail_value],
        dim=1,
    )

    def full_attention():
        prefix_scores = torch.einsum(
            "bhd,khd->bhk", query.float(), prefix_key.float()
        ) * scale
        tail_scores = torch.einsum(
            "bhd,bthd->bht", query.float(), tail_key.float()
        ) * scale
        full_scores = torch.cat([prefix_scores, tail_scores], dim=2)
        full_weights = torch.softmax(full_scores, dim=-1)
        return torch.einsum(
            "bhk,bkhd->bhd", full_weights, full_value.float()
        ).to(dtype)

    def decomposed_attention():
        prefix_scores = torch.einsum(
            "bhd,khd->bhk", query.float(), prefix_key.float()
        ) * scale
        tail_scores = torch.einsum(
            "bhd,bthd->bht", query.float(), tail_key.float()
        ) * scale
        prefix_weights = torch.softmax(prefix_scores, dim=-1)
        tail_weights = torch.softmax(tail_scores, dim=-1)
        prefix_output = torch.einsum(
            "bhk,khd->bhd", prefix_weights, prefix_value.float()
        ).to(dtype)
        tail_output = torch.einsum(
            "bht,bthd->bhd", tail_weights, tail_value.float()
        ).to(dtype)
        prefix_lse = torch.logsumexp(prefix_scores, dim=-1)
        tail_lse = torch.logsumexp(tail_scores, dim=-1)
        max_lse = torch.maximum(prefix_lse, tail_lse)
        prefix_weight = torch.exp(prefix_lse - max_lse)
        tail_weight = torch.exp(tail_lse - max_lse)
        total_weight = prefix_weight + tail_weight
        return (
            prefix_output.float() * (prefix_weight / total_weight).unsqueeze(-1)
            + tail_output.float() * (tail_weight / total_weight).unsqueeze(-1)
        ).to(dtype)

    full_output = full_attention()
    decomposed_output = decomposed_attention()
    max_abs_error = float(
        (full_output.float() - decomposed_output.float()).abs().max().item()
    )
    full_ms = _timed_call(full_attention)
    decomposed_ms = _timed_call(decomposed_attention)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "shape": {
            "batch_size": batch_size,
            "prefix_len": prefix_len,
            "tail_len": tail_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "max_abs_error": max_abs_error,
        "full_attention_average_ms": full_ms,
        "decomposed_attention_average_ms": decomposed_ms,
        "warmup_iterations": 3,
        "timed_iterations": 20,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA-compatible GPU required", file=sys.stderr)
        raise SystemExit(1)
    main()
