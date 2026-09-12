"""Measure the DSA prefill kernel's compile identities across row strides."""

import hashlib
import json
import types

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_prefill_kernel,
)


TOPK = 2048
CONTEXT_LENGTHS = (4096, 4160, 4224, 65537)


def make_pre_fix_kernel():
    current_fn = transform_index_page_table_prefill_kernel.fn
    old_fn = types.FunctionType(
        current_fn.__code__,
        current_fn.__globals__,
        current_fn.__name__,
        current_fn.__defaults__,
        current_fn.__closure__,
    )
    old_fn.__annotations__ = dict(current_fn.__annotations__)
    old_fn.__annotations__["page_table_stride_0"] = tl.constexpr
    return triton.JITFunction(old_fn)


def launch(kernel, context_length):
    extend_lens = (2, 1)
    real_tokens = sum(extend_lens)
    page_table = (
        torch.arange(context_length, dtype=torch.int32, device="cuda")[None, :]
        + torch.arange(real_tokens, dtype=torch.int32, device="cuda")[:, None] * 17
    )
    topk = (
        torch.arange(TOPK, dtype=torch.int64, device="cuda")
        .remainder(context_length)
        .repeat(real_tokens, 1)
    )
    topk[:, 0] = 0
    topk[:, 1] = context_length - 1
    topk[:, 257::257] = -1
    cu_seqlens = torch.tensor((0, 2, 3), dtype=torch.int32, device="cuda")
    result = torch.empty((real_tokens, TOPK), dtype=torch.int32, device="cuda")
    kernel[(2, 1, 8)](
        page_table,
        topk,
        cu_seqlens,
        result,
        page_table.stride(0),
        page_table.stride(1),
        topk.stride(0),
        topk.stride(1),
        result.stride(0),
        result.stride(1),
        PAGE_TABLE_IS_EXPANDED=True,
        TOPK=TOPK,
        BLOCK_Q=4,
        BLOCK_TOPK=256,
        num_warps=4,
    )
    torch.cuda.synchronize()

    expected = torch.gather(page_table, 1, topk.clamp(min=0))
    expected[topk < 0] = -1
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def summarize(kernel):
    cache = kernel.device_caches[torch.cuda.current_device()][0]
    summaries = []
    for cache_key, compiled in cache.items():
        binary = compiled.asm.get("hsaco", b"")
        if isinstance(binary, str):
            binary = binary.encode()
        summaries.append(
            {
                "cache_key": repr(cache_key),
                "compiled_name": compiled.name,
                "hsaco_bytes": len(binary),
                "hsaco_sha256": hashlib.sha256(binary).hexdigest(),
            }
        )
    return summaries


def measure(kernel):
    counts = []
    for context_length in CONTEXT_LENGTHS:
        launch(kernel, context_length)
        cache = kernel.device_caches[torch.cuda.current_device()][0]
        counts.append({"stride": context_length, "cache_entries": len(cache)})
    return {"counts": counts, "compiled_identities": summarize(kernel)}


def main():
    pre_fix = make_pre_fix_kernel()
    current = transform_index_page_table_prefill_kernel
    stride_params = {}
    for label, kernel in (("pre_fix", pre_fix), ("current", current)):
        param = next(p for p in kernel.params if p.name == "page_table_stride_0")
        stride_params[label] = {
            "is_constexpr": param.is_constexpr,
            "do_not_specialize": param.do_not_specialize,
        }

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "stride_params": stride_params,
                "pre_fix": measure(pre_fix),
                "current": measure(current),
                "numerical_reference": "torch.gather with explicit -1 masking; exact match",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
