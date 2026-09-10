import argparse
import hashlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton
from torch.profiler import ProfilerActivity, profile

from sglang.kernels.ops.attention.extend_attention import (
    _get_block_sizes_for_extend_attention,
    dense_prefill_attention_fwd,
)


FP8 = torch.float8_e4m3fn
SEQUENCE_LENGTH = 4096
QUERY_HEADS = 8
KEY_VALUE_HEADS = 2
HEAD_DIMENSION = 128
VALUE_DIMENSION = 128
KEY_SCALE = 0.5
VALUE_SCALE = 1.25
PARTITIONS = [
    [4096],
    [2048, 2048],
    [1024, 1024, 1024, 1024],
    [512] * 8,
    [256] * 16,
    [128] * 32,
]
WARM_REPETITIONS = 3
REFERENCE_BLOCK_SIZE = 512


def _sha256(tensor):
    raw_bytes = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True
    ).strip()


def _make_inputs(mode, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    def random_tensor(shape):
        return torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ) * 0.05

    query = random_tensor((SEQUENCE_LENGTH, QUERY_HEADS, HEAD_DIMENSION))
    key = random_tensor((SEQUENCE_LENGTH, KEY_VALUE_HEADS, HEAD_DIMENSION))
    value = random_tensor((SEQUENCE_LENGTH, KEY_VALUE_HEADS, VALUE_DIMENSION))
    if mode == "bf16":
        return query, key, value, 1.0, 1.0
    query_fp8 = query.to(FP8)
    key_fp8 = (key.float() / KEY_SCALE).to(FP8)
    value_fp8 = (value.float() / VALUE_SCALE).to(FP8)
    return query_fp8, key_fp8, value_fp8, KEY_SCALE, VALUE_SCALE


def _reference(query, key, value, key_scale, value_scale):
    query_float = query.float()
    key_float = key.float() * key_scale
    value_float = value.float() * value_scale
    key_expanded = key_float.repeat_interleave(
        QUERY_HEADS // KEY_VALUE_HEADS, dim=1
    )
    value_expanded = value_float.repeat_interleave(
        QUERY_HEADS // KEY_VALUE_HEADS, dim=1
    )
    output = torch.empty(
        (SEQUENCE_LENGTH, QUERY_HEADS, VALUE_DIMENSION),
        dtype=torch.float32,
        device=query.device,
    )
    softmax_scale = HEAD_DIMENSION**-0.5
    for start in range(0, SEQUENCE_LENGTH, REFERENCE_BLOCK_SIZE):
        stop = min(start + REFERENCE_BLOCK_SIZE, SEQUENCE_LENGTH)
        query_block = query_float[start:stop]
        scores = torch.einsum(
            "bhd,khe->bhk", query_block, key_expanded
        ) * softmax_scale
        query_positions = torch.arange(
            start, stop, device=query.device
        )[:, None, None]
        key_positions = torch.arange(
            SEQUENCE_LENGTH, device=query.device
        )[None, None, :]
        scores = scores.masked_fill(query_positions < key_positions, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        output[start:stop] = torch.einsum(
            "bhk,khe->bhe", probabilities, value_expanded
        )
    return output


def _run_partition(query, key, value, output, partition, key_scale, value_scale, block_m):
    grids = []
    start = 0
    for chunk_length in partition:
        stop = start + chunk_length
        query_chunk = query[start:stop]
        key_prefix = key[:stop]
        value_prefix = value[:stop]
        output_chunk = output[start:stop]
        query_indptr = torch.tensor(
            [0, chunk_length], dtype=torch.int32, device=query.device
        )
        key_value_indptr = torch.tensor(
            [0, stop], dtype=torch.int32, device=query.device
        )
        dense_prefill_attention_fwd(
            query_chunk,
            key_prefix,
            value_prefix,
            output_chunk,
            query_indptr,
            key_value_indptr,
            chunk_length,
            sm_scale=HEAD_DIMENSION**-0.5,
            k_scale=key_scale,
            v_scale=value_scale,
            is_causal=True,
        )
        grids.append(
            {
                "batch": 1,
                "heads": QUERY_HEADS,
                "query_blocks": triton.cdiv(chunk_length, block_m),
            }
        )
        start = stop
    torch.cuda.synchronize()
    return grids


def _profile_dispatch(mode):
    query, key, value, key_scale, value_scale = _make_inputs(mode, 291954)
    output = torch.empty(
        (64, QUERY_HEADS, VALUE_DIMENSION),
        dtype=torch.bfloat16,
        device="cuda",
    )
    query_indptr = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    key_value_indptr = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        dense_prefill_attention_fwd(
            query,
            key,
            value,
            output,
            query_indptr,
            key_value_indptr,
            64,
            sm_scale=HEAD_DIMENSION**-0.5,
            k_scale=key_scale,
            v_scale=value_scale,
            is_causal=True,
        )
        torch.cuda.synchronize()
    events = [
        event
        for event in profiler.key_averages()
        if "dense_prefill" in event.key
    ]
    if not events:
        raise RuntimeError("Profiler did not capture the dense prefill kernel")
    event = events[0]
    return {
        "kernel_name": event.key,
        "device_time_microseconds": event.device_time,
        "count": event.count,
    }


def _measure_case(mode, query, key, value, key_scale, value_scale, partition, block_m):
    output = torch.empty(
        (SEQUENCE_LENGTH, QUERY_HEADS, VALUE_DIMENSION),
        dtype=torch.bfloat16,
        device="cuda",
    )
    event_times = []
    wall_times = []
    peak_memory_bytes = []
    allocated_before_bytes = []
    allocated_after_bytes = []
    reserved_after_bytes = []
    allocation_retries = []
    out_of_memory_syncs = []
    grids = None
    _run_partition(
        query,
        key,
        value,
        output,
        partition,
        key_scale,
        value_scale,
        block_m,
    )
    for _ in range(WARM_REPETITIONS):
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        grids = _run_partition(
            query,
            key,
            value,
            output,
            partition,
            key_scale,
            value_scale,
            block_m,
        )
        end_event.record()
        torch.cuda.synchronize()
        wall_stop = time.perf_counter()
        event_times.append(start_event.elapsed_time(end_event) / 1000.0)
        wall_times.append(wall_stop - wall_start)
        memory_stats = torch.cuda.memory_stats()
        peak_memory_bytes.append(torch.cuda.max_memory_allocated())
        allocated_before_bytes.append(allocated_before)
        allocated_after_bytes.append(torch.cuda.memory_allocated())
        reserved_after_bytes.append(torch.cuda.memory_reserved())
        allocation_retries.append(memory_stats.get("num_alloc_retries", 0))
        out_of_memory_syncs.append(memory_stats.get("num_oom_sync", 0))
    key_value_visits = sum(
        sum(partition[: index + 1]) for index in range(len(partition))
    )
    mean_event_time = statistics.mean(event_times)
    return {
        "partition": partition,
        "partition_count": len(partition),
        "query_tokens": SEQUENCE_LENGTH,
        "key_value_visits": key_value_visits,
        "event_times_seconds": event_times,
        "wall_times_seconds": wall_times,
        "mean_event_time_seconds": mean_event_time,
        "median_event_time_seconds": statistics.median(event_times),
        "query_tokens_per_second": SEQUENCE_LENGTH / mean_event_time,
        "key_value_visits_per_second": key_value_visits / mean_event_time,
        "peak_memory_bytes": max(peak_memory_bytes),
        "allocated_before_bytes": allocated_before_bytes[0],
        "allocated_after_bytes": allocated_after_bytes[-1],
        "reserved_after_bytes": reserved_after_bytes[-1],
        "allocation_retries": max(allocation_retries),
        "out_of_memory_syncs": max(out_of_memory_syncs),
        "grids": grids,
        "output_sha256": _sha256(output),
        "final_token_sha256": _sha256(output[-1]),
        "output_tensor": output,
    }


def _cold_compile(mode):
    query, key, value, key_scale, value_scale = _make_inputs(mode, 291955)
    output = torch.empty(
        (64, QUERY_HEADS, VALUE_DIMENSION),
        dtype=torch.bfloat16,
        device="cuda",
    )
    query_indptr = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    key_value_indptr = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    dense_prefill_attention_fwd(
        query,
        key,
        value,
        output,
        query_indptr,
        key_value_indptr,
        64,
        sm_scale=HEAD_DIMENSION**-0.5,
        k_scale=key_scale,
        v_scale=value_scale,
        is_causal=True,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", default="unavailable")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This benchmark requires exactly one GPU")
    device_properties = torch.cuda.get_device_properties(0)
    if device_properties.gcnArchName.split(":")[0] != "gfx942":
        raise RuntimeError("This benchmark requires gfx942")
    block_sizes = _get_block_sizes_for_extend_attention(
        HEAD_DIMENSION, VALUE_DIMENSION
    )
    block_m = block_sizes[3]
    results = {
        "label": "gfx942 dense prefill chunk-partition benchmark",
        "source_commit": _git_commit(),
        "image_identity": args.image_id,
        "command": " ".join(sys.argv),
        "gpu": {
            "name": device_properties.name,
            "gcn_arch": device_properties.gcnArchName,
            "capability": list(torch.cuda.get_device_capability(0)),
            "uuid": str(device_properties.uuid),
            "total_memory_bytes": device_properties.total_memory,
        },
        "stack": {
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_path": torch.__file__,
            "triton": triton.__version__,
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "kernel_module": dense_prefill_attention_fwd.__module__,
            "kernel_source": inspect.getsourcefile(dense_prefill_attention_fwd),
        },
        "workload": {
            "sequence_length": SEQUENCE_LENGTH,
            "query_heads": QUERY_HEADS,
            "key_value_heads": KEY_VALUE_HEADS,
            "head_dimension": HEAD_DIMENSION,
            "value_dimension": VALUE_DIMENSION,
            "partitions": PARTITIONS,
            "warm_repetitions": WARM_REPETITIONS,
            "reference_block_size": REFERENCE_BLOCK_SIZE,
            "key_scale": KEY_SCALE,
            "value_scale": VALUE_SCALE,
            "block_m": block_m,
            "block_n": block_sizes[4],
            "num_warps": block_sizes[5],
        },
        "numerical_gates": {
            "bf16_reference": {"rtol": 2e-2, "atol": 2e-2},
            "fp8_reference": {"rtol": 5e-2, "atol": 5e-2},
            "chunked_vs_unchunked": "same gate as the corresponding dtype",
        },
        "modes": {},
    }
    for mode in ("bf16", "fp8"):
        query, key, value, key_scale, value_scale = _make_inputs(mode, 291953)
        reference = _reference(query, key, value, key_scale, value_scale)
        cold_compile_seconds = _cold_compile(mode)
        dispatch = _profile_dispatch(mode)
        mode_results = {
            "cold_compile_seconds": cold_compile_seconds,
            "dispatch": dispatch,
            "input_bytes": {
                "query": query.numel() * query.element_size(),
                "key": key.numel() * key.element_size(),
                "value": value.numel() * value.element_size(),
            },
            "output_bytes": SEQUENCE_LENGTH
            * QUERY_HEADS
            * VALUE_DIMENSION
            * torch.empty((), dtype=torch.bfloat16).element_size(),
            "cases": [],
        }
        unchunked_output = None
        for partition in PARTITIONS:
            case = _measure_case(
                mode,
                query,
                key,
                value,
                key_scale,
                value_scale,
                partition,
                block_m,
            )
            output_tensor = case.pop("output_tensor")
            gate = (
                results["numerical_gates"]["bf16_reference"]
                if mode == "bf16"
                else results["numerical_gates"]["fp8_reference"]
            )
            torch.testing.assert_close(
                output_tensor.float(), reference, rtol=gate["rtol"], atol=gate["atol"]
            )
            if unchunked_output is None:
                unchunked_output = output_tensor
            else:
                torch.testing.assert_close(
                    output_tensor.float(),
                    unchunked_output.float(),
                    rtol=gate["rtol"],
                    atol=gate["atol"],
                )
            case["reference_max_abs_error"] = float(
                (output_tensor.float() - reference).abs().max()
            )
            case["reference_mean_abs_error"] = float(
                (output_tensor.float() - reference).abs().mean()
            )
            case["chunked_vs_unchunked_max_abs_error"] = float(
                (output_tensor.float() - unchunked_output.float()).abs().max()
            )
            case["chunked_vs_unchunked_mean_abs_error"] = float(
                (output_tensor.float() - unchunked_output.float()).abs().mean()
            )
            case["final_token_vs_unchunked_max_abs_error"] = float(
                (output_tensor[-1].float() - unchunked_output[-1].float()).abs().max()
            )
            mode_results["cases"].append(case)
        results["modes"][mode] = mode_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
