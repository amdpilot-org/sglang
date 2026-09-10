"""Bounded Aiter sparse-MLA compiled-path reuse benchmark on MI300X."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-benchmark", runner_config="amd")


SENTINEL = -31337.0
BATCHES = (1, 2, 4)
CONTEXT_TOKENS = 128
TOPK = 64
NUM_HEADS = 16
D_QK = 576
D_V = 512
NUM_KV_SPLITS = 4


def _device_supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and "MI300X" in torch.cuda.get_device_name(0)
    )


def _reference(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    selected = kv.view(-1, D_QK)[indices].float().view(-1, TOPK, D_QK)
    scores = torch.einsum("bhd,btd->bht", q.float(), selected) * (D_QK**-0.5)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,btd->bhd", probabilities, selected[..., :D_V])


def _native_module_paths() -> dict[str, str | None]:
    names = (
        "aiter.jit.module_aiter_core",
        "aiter.jit.module_mla_metadata",
        "aiter.jit.module_mla_reduce",
    )
    return {name: getattr(sys.modules.get(name), "__file__", None) for name in names}


def _profiled_dispatch(call, *args, **kwargs) -> list[dict[str, Any]]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        call(*args, **kwargs)
        torch.cuda.synchronize()
    return [
        {
            "name": event.key,
            "count": event.count,
            "self_device_time_us": event.self_device_time_total,
        }
        for event in prof.key_averages()
        if event.device_type == torch.profiler.DeviceType.CUDA
        and event.self_device_time_total > 0
    ]


def run() -> dict[str, Any]:
    if not _device_supported():
        raise RuntimeError("This benchmark requires one AMD Instinct MI300X (gfx942)")

    import aiter
    from aiter.mla import mla_decode_fwd

    device = torch.device("cuda")
    torch.manual_seed(34947)
    metadata_info = aiter.get_mla_metadata_info_v1(
        max(BATCHES),
        1,
        NUM_HEADS,
        torch.bfloat16,
        torch.bfloat16,
        is_sparse=True,
        fast_mode=True,
        num_kv_splits=NUM_KV_SPLITS,
    )
    metadata = [
        torch.empty(size, dtype=dtype, device=device) for size, dtype in metadata_info
    ]
    (
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = metadata
    metadata_addresses = [tensor.data_ptr() for tensor in metadata]
    results: list[dict[str, Any]] = []

    for batch in BATCHES:
        q = (
            torch.randn(
                batch,
                NUM_HEADS,
                D_QK,
                device=device,
                dtype=torch.bfloat16,
            )
            / 10
        ).contiguous()
        kv = (
            (
                torch.randn(
                    batch * CONTEXT_TOKENS,
                    1,
                    D_QK,
                    device=device,
                    dtype=torch.bfloat16,
                )
                / 10
            )
            .contiguous()
            .view(batch * CONTEXT_TOKENS, 1, 1, D_QK)
        )
        qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
        kv_indptr = torch.arange(
            0,
            (batch + 1) * CONTEXT_TOKENS,
            CONTEXT_TOKENS,
            dtype=torch.int32,
            device=device,
        )
        kv_last_page_lens = torch.ones(batch, dtype=torch.int32, device=device)
        indices = (
            torch.stack(
                [
                    torch.randperm(CONTEXT_TOKENS, device=device, dtype=torch.int32)[
                        :TOPK
                    ]
                    for _ in range(batch)
                ]
            )
            .reshape(-1)
            .contiguous()
        )
        aiter.get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_lens,
            NUM_HEADS,
            1,
            True,
            *metadata,
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=True,
            max_split_per_batch=NUM_KV_SPLITS,
            topk=TOPK,
            dtype_q_nope=torch.bfloat16,
            dtype_kv_nope=torch.bfloat16,
        )
        output = torch.empty(batch, NUM_HEADS, D_V, device=device, dtype=torch.bfloat16)
        output_address = output.data_ptr()
        reference = _reference(q, kv, indices)
        call_kwargs = {
            "page_size": 1,
            "nhead_kv": 1,
            "sm_scale": D_QK**-0.5,
            "num_kv_splits": NUM_KV_SPLITS,
            "num_kv_splits_indptr": None,
            "work_meta_data": work_metadata,
            "work_indptr": work_indptr,
            "work_info_set": work_info_set,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
            "causal": False,
        }

        for phase in ("cold", "warm"):
            output.fill_(SENTINEL)
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            host_start = time.perf_counter()
            logits, _ = mla_decode_fwd(
                q,
                kv,
                output,
                qo_indptr,
                kv_indptr,
                indices,
                kv_last_page_lens,
                1,
                **call_kwargs,
            )
            host_elapsed = time.perf_counter() - host_start
            end_event.record()
            torch.cuda.synchronize()
            device_elapsed = start_event.elapsed_time(end_event) / 1000.0
            torch.testing.assert_close(
                output.float(),
                reference,
                atol=5e-4,
                rtol=5e-3,
            )
            assert not torch.any(output == SENTINEL)
            assert output.data_ptr() == output_address
            assert logits.dtype == torch.float32
            assert logits.data_ptr() != output.data_ptr()
            assert [tensor.data_ptr() for tensor in metadata] == metadata_addresses
            results.append(
                {
                    "batch": batch,
                    "phase": phase,
                    "host_elapsed_seconds": host_elapsed,
                    "device_elapsed_seconds": device_elapsed,
                    "max_abs_error": (output.float() - reference).abs().max().item(),
                    "sentinel_remaining": bool(torch.any(output == SENTINEL)),
                    "output_address": output.data_ptr(),
                    "metadata_addresses_stable": [
                        tensor.data_ptr() == address
                        for tensor, address in zip(metadata, metadata_addresses)
                    ],
                    "dtypes": {
                        "q": str(q.dtype),
                        "kv": str(kv.dtype),
                        "output": str(output.dtype),
                        "logits": str(logits.dtype),
                    },
                    "logits_aliases_output": logits.data_ptr() == output.data_ptr(),
                }
            )

    unsupported_error = ""
    unsupported_q = torch.zeros(1, 24, D_QK, device=device, dtype=torch.bfloat16)
    unsupported_kv = torch.zeros(1, 1, 1, D_QK, device=device, dtype=torch.bfloat16)
    unsupported_output = torch.empty(1, 24, D_V, device=device, dtype=torch.bfloat16)
    unsupported_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    unsupported_indices = torch.tensor([0], dtype=torch.int32, device=device)
    try:
        mla_decode_fwd(
            unsupported_q,
            unsupported_kv,
            unsupported_output,
            unsupported_indptr,
            unsupported_indptr,
            unsupported_indices,
            unsupported_indptr[:1],
            1,
            page_size=1,
            nhead_kv=1,
            sm_scale=D_QK**-0.5,
            work_meta_data=work_metadata,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )
    except AssertionError as error:
        unsupported_error = str(error)
    else:
        raise AssertionError("Unsupported 24-head variant unexpectedly dispatched")

    dispatch = _profiled_dispatch(
        mla_decode_fwd,
        q,
        kv,
        output,
        qo_indptr,
        kv_indptr,
        indices,
        kv_last_page_lens,
        1,
        **call_kwargs,
    )
    return {
        "operation": "aiter.mla.mla_decode_fwd sparse persistent BF16 path",
        "shape_sequence": {
            "batches": list(BATCHES),
            "context_tokens": CONTEXT_TOKENS,
            "topk": TOPK,
            "heads": NUM_HEADS,
            "d_qk": D_QK,
            "d_v": D_V,
            "kv_splits": NUM_KV_SPLITS,
        },
        "timing_matrix": results,
        "native_dispatch": dispatch,
        "native_modules": _native_module_paths(),
        "unsupported_variant": {
            "heads": 24,
            "error": unsupported_error,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
