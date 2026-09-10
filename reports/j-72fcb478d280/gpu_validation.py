"""Bounded gfx942 validation for the Qwen4 PLE fused hash/gather candidate."""

from __future__ import annotations

import argparse
import json

import torch
import triton.testing

from sglang.kernels.ops.qwen4_ple import (
    fused_qwen4_ngram_hash,
)
from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel


HEADS = 16
EMBEDDING_DIM = 160
ROWS_PER_HEAD = 65537
TP_SIZE = 8
BENCHMARK_REP_MS = 20


def reference_ids(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    previous = torch.where(
        (contexts[:, 0] == 0) | (contexts[:, 1] == 0), 0, contexts[:, 0]
    )
    mixed = (contexts[:, 2] * multipliers[0]) ^ (contexts[:, 1] * multipliers[1])
    mixed_three = mixed ^ (previous * multipliers[2])
    return torch.cat(
        (
            mixed[:, None].remainder(sizes[:8]) + offsets[:8],
            mixed_three[:, None].remainder(sizes[8:]) + offsets[8:],
        ),
        dim=1,
    )


def timed_call(function) -> tuple[float, torch.Tensor]:
    result = function()
    torch.cuda.synchronize()
    mean_ms = triton.testing.do_bench_cudagraph(
        function, rep=BENCHMARK_REP_MS, return_mode="mean"
    )
    return mean_ms, result



def validate_and_time(dtype: torch.dtype, tokens: int) -> dict[str, object]:
    torch.manual_seed(38731 + tokens)
    contexts = torch.randint(
        0, 151000, (tokens, 3), dtype=torch.long, device="cuda"
    )
    multipliers = torch.tensor(
        [190734863281251, 953674316406251, 4768371582031251],
        dtype=torch.long,
        device="cuda",
    )
    sizes = torch.full((HEADS,), ROWS_PER_HEAD, dtype=torch.long, device="cuda")
    offsets = torch.arange(HEADS, dtype=torch.long, device="cuda") * ROWS_PER_HEAD
    local_rows = (HEADS * ROWS_PER_HEAD + TP_SIZE - 1) // TP_SIZE
    weight = torch.empty((local_rows, EMBEDDING_DIM), dtype=dtype, pin_memory=True)
    weight.copy_(
        (
            torch.arange(weight.numel()).reshape(weight.shape) % 31 - 15
        ).to(dtype)
    )
    output = torch.empty(
        (tokens, HEADS, EMBEDDING_DIM), dtype=torch.bfloat16, device="cuda"
    )

    def hash_and_gather() -> torch.Tensor:
        ids = fused_qwen4_ngram_hash(
            contexts, multipliers, sizes, offsets, 0
        )
        _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
            weight.data_ptr(),
            ids,
            output,
            embedding_dim=EMBEDDING_DIM,
            tp_vocab_start=0,
            tp_vocab_end=local_rows,
            is_fp8=dtype == torch.float8_e4m3fn,
            BLOCK_D=256,
        )
        return output

    def fused() -> torch.Tensor:
        return fused_qwen4_ngram_gather(
            contexts,
            multipliers,
            sizes,
            offsets,
            0,
            weight,
            0,
            local_rows,
            output,
        )

    gather_ms, gather_output = timed_call(hash_and_gather)

    cpu_contexts = contexts.cpu()
    cpu_ids = reference_ids(
        cpu_contexts, multipliers.cpu(), sizes.cpu(), offsets.cpu()
    )
    in_range = (cpu_ids >= 0) & (cpu_ids < local_rows)
    local_ids = torch.where(in_range, cpu_ids, 0)
    cpu_rows = weight.to(dtype=torch.bfloat16)
    expected = torch.where(
        in_range[..., None], cpu_rows[local_ids], torch.zeros(())
    ).to(device="cuda")
    torch.testing.assert_close(gather_output, expected, rtol=0, atol=0)

    table_item_bytes = 1 if dtype == torch.float8_e4m3fn else 2
    effective_bytes = (
        tokens
        * HEADS
        * EMBEDDING_DIM
        * (table_item_bytes + 2)
        + tokens
        * HEADS
        * 8
        * 2
    )
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "tokens": tokens,
        "local_rows": local_rows,
        "table_bytes": weight.numel() * weight.element_size(),
        "benchmark_rep_ms": BENCHMARK_REP_MS,
        "hash_gather_ms": gather_ms,
        "effective_gbps": effective_bytes / (gather_ms / 1000.0) / 1e9,
        "reference_max_abs_error": int(
            (gather_output.float() - expected.float()).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    results = []
    for dtype in (torch.float8_e4m3fn, torch.bfloat16):
        for tokens in (1, 16, 64, 256, 1024):
            result = validate_and_time(dtype, tokens)
            results.append(result)
            print(json.dumps(result, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "results": results,
                },
                file,
                indent=2,
                sort_keys=True,
            )
            file.write("\n")


if __name__ == "__main__":
    main()
