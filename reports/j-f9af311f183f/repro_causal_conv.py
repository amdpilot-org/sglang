import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F


sys.path.insert(0, os.environ.get("SGLANG_SOURCE", "/job/sglang/python"))

from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update


def _pattern(shape, device, dtype, offset=0):
    count = math.prod(shape)
    values = torch.arange(count, device=device, dtype=torch.float32)
    values = ((values + offset) % 11) - 5
    return (values * 0.37).reshape(shape).to(dtype).contiguous()


def independent_reference(x, conv_state, weight, bias, cache_seqlens):
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    state_ref = conv_state.clone()
    if cache_seqlens is None:
        history = conv_state[..., -(width - 1) :]
        x_new = torch.cat([history, x], dim=-1).to(weight.dtype)
        state_ref.copy_(torch.cat([conv_state, x], dim=-1)[..., -state_len:])
    else:
        history_idx = (
            cache_seqlens.to(torch.long)[:, None]
            - (width - 1)
            + torch.arange(width - 1, device=x.device, dtype=torch.long)[None, :]
        ) % state_len
        history_idx = history_idx[:, None, :].expand(batch, dim, width - 1)
        history = conv_state.gather(2, history_idx)
        x_new = torch.cat([history, x], dim=-1).to(weight.dtype)
        copy_idx = (
            cache_seqlens.to(torch.long)[:, None]
            + torch.arange(seqlen, device=x.device, dtype=torch.long)[None, :]
        ) % state_len
        copy_idx = copy_idx[:, None, :].expand(batch, dim, seqlen)
        state_ref.scatter_(2, copy_idx, x)
    out = F.conv1d(
        x_new,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=dim,
    )[..., -seqlen:]
    out = F.silu(out).to(x.dtype)
    return out, state_ref


def run_chunked(x, state, weight, bias, cache_seqlens):
    state_actual = state.clone()
    out = causal_conv1d_update(
        x.clone(),
        state_actual,
        weight,
        bias,
        activation="silu",
        cache_seqlens=cache_seqlens,
    )
    return out, state_actual


def run_recurrence(x, state, weight, bias, cache_seqlens):
    state_actual = state.clone()
    cursor = cache_seqlens.clone()
    outputs = []
    for token in range(x.shape[-1]):
        token_x = x[..., token : token + 1].contiguous()
        output = causal_conv1d_update(
            token_x,
            state_actual,
            weight,
            bias,
            activation="silu",
            cache_seqlens=cursor,
        )
        outputs.append(output)
        cursor = (cursor + 1) % state.shape[-1]
    return torch.cat(outputs, dim=-1), state_actual


def compare(actual, expected, rtol, atol):
    if actual.shape != expected.shape:
        return False, float("inf"), float("inf")
    difference = (actual.float() - expected.float()).abs()
    return (
        torch.allclose(actual, expected, rtol=rtol, atol=atol),
        difference.max().item(),
        difference.mean().item(),
    )


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(0)
    cases = [
        {"name": "w2_l4_cursor3", "width": 2, "state_len": 4, "cursor": 3, "chunks": [2, 3, 4], "dtype": torch.float32},
        {"name": "w3_l4_cursor2", "width": 3, "state_len": 4, "cursor": 2, "chunks": [2, 3, 4], "dtype": torch.float32},
        {"name": "w4_l4_cursor1", "width": 4, "state_len": 4, "cursor": 1, "chunks": [2, 3, 4], "dtype": torch.float32},
        {"name": "w4_l8_cursor7", "width": 4, "state_len": 8, "cursor": 7, "chunks": [3, 4, 5], "dtype": torch.float32},
        {"name": "w3_l8_cursor5", "width": 3, "state_len": 8, "cursor": 5, "chunks": [2, 3, 4], "dtype": torch.float32},
        {"name": "w2_l8_cursor6", "width": 2, "state_len": 8, "cursor": 6, "chunks": [3, 4, 5], "dtype": torch.float32},
        {"name": "w4_l4_cursor3_bf16", "width": 4, "state_len": 4, "cursor": 3, "chunks": [2, 3, 4], "dtype": torch.bfloat16},
        {"name": "w3_l8_cursor7_bf16", "width": 3, "state_len": 8, "cursor": 7, "chunks": [2, 3, 4], "dtype": torch.bfloat16},
    ]
    results = []
    start = time.perf_counter()
    for case_index, case in enumerate(cases):
        batch, dim = 2, 7
        total_tokens = sum(case["chunks"])
        dtype = case["dtype"]
        rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (1e-2, 5e-2)
        x = _pattern((batch, dim, total_tokens), device, dtype, case_index * 13)
        state = _pattern((batch, dim, case["state_len"]), device, dtype, case_index * 29)
        weight = _pattern((dim, case["width"]), device, dtype, case_index * 41)
        bias = _pattern((dim,), device, dtype, case_index * 53)
        cache = torch.full((batch,), case["cursor"], device=device, dtype=torch.int32)
        chunk_outputs = []
        recurrence_outputs = []
        reference_outputs = []
        chunk_state = state.clone()
        recurrence_state = state.clone()
        reference_state = state.clone()
        chunk_cursor = cache.clone()
        recurrence_cursor = cache.clone()
        reference_cursor = cache.clone()
        token_offset = 0
        for chunk_len in case["chunks"]:
            chunk_x = x[..., token_offset : token_offset + chunk_len].contiguous()
            chunk_out, chunk_state = run_chunked(
                chunk_x, chunk_state, weight, bias, chunk_cursor
            )
            recurrence_out, recurrence_state = run_recurrence(
                chunk_x, recurrence_state, weight, bias, recurrence_cursor
            )
            reference_out, reference_state = independent_reference(
                chunk_x, reference_state, weight, bias, reference_cursor
            )
            chunk_outputs.append(chunk_out)
            recurrence_outputs.append(recurrence_out)
            reference_outputs.append(reference_out)
            chunk_cursor = (chunk_cursor + chunk_len) % case["state_len"]
            recurrence_cursor = (recurrence_cursor + chunk_len) % case["state_len"]
            reference_cursor = (reference_cursor + chunk_len) % case["state_len"]
            token_offset += chunk_len
        chunk_out = torch.cat(chunk_outputs, dim=-1)
        recurrence_out = torch.cat(recurrence_outputs, dim=-1)
        reference_out = torch.cat(reference_outputs, dim=-1)
        chunk_vs_reference = compare(chunk_out, reference_out, rtol, atol)
        recurrence_vs_reference = compare(recurrence_out, reference_out, rtol, atol)
        chunk_vs_recurrence = compare(chunk_out, recurrence_out, rtol, atol)
        chunk_state_vs_reference = compare(chunk_state, reference_state, 0, 0)
        recurrence_state_vs_reference = compare(recurrence_state, reference_state, 0, 0)
        results.append(
            {
                "name": case["name"],
                "width": case["width"],
                "state_len": case["state_len"],
                "initial_cursor": case["cursor"],
                "chunks": case["chunks"],
                "dtype": str(dtype).replace("torch.", ""),
                "rtol": rtol,
                "atol": atol,
                "chunk_vs_reference": chunk_vs_reference,
                "recurrence_vs_reference": recurrence_vs_reference,
                "chunk_vs_recurrence": chunk_vs_recurrence,
                "chunk_state_vs_reference": chunk_state_vs_reference,
                "recurrence_state_vs_reference": recurrence_state_vs_reference,
            }
        )
    elapsed = time.perf_counter() - start
    report = {
        "source": os.path.join(
            os.environ.get("SGLANG_SOURCE", "/job/sglang/python"),
            "sglang/kernels/ops/mamba/causal_conv1d_triton.py",
        ),
        "source_commit": os.popen(
            "git -C "
            + os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__import__("sglang").__file__)))
            )
            + " rev-parse HEAD"
        ).read().strip(),
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": elapsed,
        "cases": results,
    }
    output_path = os.environ.get("REPRO_OUTPUT", "/job/repro-results.json")
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
