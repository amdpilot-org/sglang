import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.speculative.dflash_info import DFlashVerifyInput


BATCH_SIZES = [1, 8, 1, 16, 1, 32, 1]
DRAFT_TOKEN_NUM = 8
PAGED_KERNEL_LEN = 10
REPEATS = 3
SENTINEL = -1


def _make_input(device):
    return DFlashVerifyInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=device),
        positions=torch.empty((0,), dtype=torch.int64, device=device),
        draft_token_num=DRAFT_TOKEN_NUM,
        custom_mask=None,
    )


def _make_batch(batch_size, device):
    req_pool_indices = torch.arange(
        batch_size, dtype=torch.int32, device=device
    )
    paged_kernel_lens = torch.full(
        (batch_size,), PAGED_KERNEL_LEN, dtype=torch.int32, device=device
    )
    paged_kernel_lens_sum = PAGED_KERNEL_LEN * batch_size
    req_to_token = torch.arange(
        batch_size * 20, dtype=torch.int32, device=device
    ).reshape(batch_size, 20)
    return req_pool_indices, paged_kernel_lens, paged_kernel_lens_sum, req_to_token


def _expected_outputs(
    batch_size,
    paged_kernel_lens,
    paged_kernel_lens_sum,
    req_to_token,
    device,
):
    expected_kv_indices = torch.cat(
        [
            req_to_token[index, : PAGED_KERNEL_LEN + DRAFT_TOKEN_NUM]
            for index in range(batch_size)
        ]
    )
    expected_cum_kv_seq_len = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=device
    )
    expected_cum_kv_seq_len[1:] = torch.cumsum(
        paged_kernel_lens + DRAFT_TOKEN_NUM, dim=0
    )
    expected_qo_indptr = torch.arange(
        0,
        (batch_size + 1) * DRAFT_TOKEN_NUM,
        DRAFT_TOKEN_NUM,
        dtype=torch.int32,
        device=device,
    )
    expected_numel = paged_kernel_lens_sum + DRAFT_TOKEN_NUM * batch_size
    return (
        expected_kv_indices,
        expected_cum_kv_seq_len,
        expected_qo_indptr,
        expected_numel,
    )


def _validate_reuse_buffer(buffer, expected_numel, req_to_token):
    if buffer.dtype != torch.int32:
        raise ValueError(
            f"kv_indices_buf must be torch.int32; got {buffer.dtype}"
        )
    if buffer.numel() < expected_numel:
        raise ValueError(
            f"kv_indices_buf has {buffer.numel()} elements; "
            f"at least {expected_numel} are required"
        )
    if buffer.device != req_to_token.device:
        raise ValueError(
            f"kv_indices_buf device {buffer.device} does not match "
            f"req_to_token device {req_to_token.device}"
        )
    buffer_start = buffer.data_ptr()
    buffer_end = buffer_start + buffer.numel() * buffer.element_size()
    input_start = req_to_token.data_ptr()
    input_end = input_start + req_to_token.numel() * req_to_token.element_size()
    if buffer_start < input_end and input_start < buffer_end:
        raise ValueError("kv_indices_buf must not alias req_to_token")


def _call(spec_input, batch, buffer=None):
    req_pool_indices, paged_kernel_lens, paged_kernel_lens_sum, req_to_token = batch
    expected_numel = paged_kernel_lens_sum + DRAFT_TOKEN_NUM * len(req_pool_indices)
    if buffer is not None:
        _validate_reuse_buffer(buffer, expected_numel, req_to_token)
    start = time.perf_counter()
    kv_indices, cum_kv_seq_len, qo_indptr, _ = spec_input.generate_attn_arg_prefill(
        req_pool_indices,
        paged_kernel_lens,
        paged_kernel_lens_sum,
        req_to_token,
        kv_indices_buf=buffer,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return kv_indices, cum_kv_seq_len, qo_indptr, elapsed_ms


def _compare(actual, expected, name):
    return {
        "name": name,
        "equal": bool(torch.equal(actual, expected)),
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype).replace("torch.", ""),
        "expected_dtype": str(expected.dtype).replace("torch.", ""),
    }


def _record_native_dispatch(device):
    spec_input = _make_input(device)
    batch = _make_batch(4, device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as profiler:
        _call(spec_input, batch)
    events = [
        event.key
        for event in profiler.key_averages()
        if "create_flashinfer_kv_indices" in event.key
    ]
    return {
        "kernel_names": events,
        "source_path": (
            "python/sglang/kernels/ops/kvcache/kv_indices.py"
        ),
    }


def _unsupported_variants(device):
    spec_input = _make_input(device)
    batch = _make_batch(2, device)
    req_pool_indices, paged_kernel_lens, paged_kernel_lens_sum, req_to_token = batch
    expected_numel = paged_kernel_lens_sum + DRAFT_TOKEN_NUM * len(req_pool_indices)
    results = {}

    wrong_dtype = torch.full((expected_numel,), SENTINEL, dtype=torch.int64, device=device)
    try:
        _validate_reuse_buffer(wrong_dtype, expected_numel, req_to_token)
        results["wrong_dtype"] = {"rejected": False}
    except ValueError as error:
        results["wrong_dtype"] = {"rejected": True, "error": str(error)}

    too_small = torch.full((expected_numel - 1,), SENTINEL, dtype=torch.int32, device=device)
    try:
        _validate_reuse_buffer(too_small, expected_numel, req_to_token)
        results["too_small"] = {"rejected": False}
    except ValueError as error:
        results["too_small"] = {"rejected": True, "error": str(error)}

    aliasing = req_to_token.view(-1)
    try:
        _validate_reuse_buffer(aliasing, expected_numel, req_to_token)
        results["aliasing"] = {"rejected": False}
    except ValueError as error:
        results["aliasing"] = {"rejected": True, "error": str(error)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    spec_input = _make_input(device)
    max_numel = max(
        PAGED_KERNEL_LEN * batch_size + DRAFT_TOKEN_NUM * batch_size
        for batch_size in BATCH_SIZES
    )
    reuse_buffer = torch.full(
        (max_numel * 2,), SENTINEL, dtype=torch.int32, device=device
    )
    reuse_address = reuse_buffer.data_ptr()

    results = {
        "label": "DFlash kv_indices fresh output versus documented safe buffer reuse",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "architecture": "gfx942",
            "device_count": torch.cuda.device_count(),
        },
        "batch_sizes": BATCH_SIZES,
        "draft_token_num": DRAFT_TOKEN_NUM,
        "paged_kernel_len": PAGED_KERNEL_LEN,
        "repeats": REPEATS,
        "sentinel": SENTINEL,
        "native_dispatch": _record_native_dispatch(device),
        "unsupported_variants": _unsupported_variants(device),
        "cases": [],
    }

    for representation in ["fresh", "reuse"]:
        retained_outputs = []
        for batch_size in BATCH_SIZES:
            batch = _make_batch(batch_size, device)
            req_pool_indices, paged_kernel_lens, paged_kernel_lens_sum, req_to_token = batch
            expected_kv_indices, expected_cum, expected_qo, expected_numel = (
                _expected_outputs(
                    batch_size,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    req_to_token,
                    device,
                )
            )
            timings = []
            output_addresses = []
            for _ in range(REPEATS):
                if representation == "reuse":
                    reuse_buffer.fill_(SENTINEL)
                    output, cum, qo, elapsed = _call(
                        spec_input, batch, buffer=reuse_buffer
                    )
                    output = output[:expected_numel]
                    sentinel_tail_intact = bool(
                        torch.equal(
                            reuse_buffer[expected_numel:],
                            torch.full_like(
                                reuse_buffer[expected_numel:], SENTINEL
                            ),
                        )
                    )
                else:
                    output, cum, qo, elapsed = _call(spec_input, batch)
                    sentinel_tail_intact = None
                retained_outputs.append(output)
                timings.append(elapsed)
                output_addresses.append(output.data_ptr())
                comparisons = [
                    _compare(output, expected_kv_indices, "kv_indices"),
                    _compare(cum, expected_cum, "cum_kv_seq_len"),
                    _compare(qo, expected_qo, "qo_indptr"),
                ]
                if not all(item["equal"] for item in comparisons):
                    raise AssertionError(
                        f"reference mismatch for {representation} batch {batch_size}: "
                        f"{comparisons}"
                    )
            results["cases"].append(
                {
                    "representation": representation,
                    "batch_size": batch_size,
                    "expected_numel": expected_numel,
                    "timings_ms": timings,
                    "timing_mean_ms": statistics.mean(timings),
                    "timing_median_ms": statistics.median(timings),
                    "output_addresses": output_addresses,
                    "output_address_stable": (
                        len(set(output_addresses)) == 1
                    ),
                    "reuse_buffer_address": (
                        reuse_address if representation == "reuse" else None
                    ),
                    "sentinel_tail_intact": sentinel_tail_intact,
                    "reference_comparisons": comparisons,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
