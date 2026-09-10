#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
    make_loc_fn,
    run_dense_fixture_eager,
)


class NoopTest:
    pass


WORKLOADS = (
    {
        "name": "decode_b1_c127",
        "prefix_lens": (127,),
        "max_context_len": 160,
    },
    {
        "name": "decode_b4_c127_to_c130",
        "prefix_lens": (127, 128, 129, 130),
        "max_context_len": 160,
    },
    {
        "name": "decode_b8_c63_to_c70",
        "prefix_lens": (63, 64, 65, 66, 67, 68, 69, 70),
        "max_context_len": 96,
    },
)

NUM_HEADS = 4
NUM_KV_HEADS = 4
HEAD_DIM = 64
HIDDEN_SIZE = 256
PAGE_SIZE = 16
RECURRENT_STEPS = 3
WARMUP_FORWARDS = 2
TIMED_FORWARDS = 10
MAX_LIVE_BYTES = 48 * 1024**3
FP8_K_SCALE = 0.03125
FP8_V_SCALE = 0.0625
NUMERICAL_ATOL = 3e-2
NUMERICAL_RTOL = 3e-2


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": dtype_name(tensor.dtype),
        "stride": list(tensor.stride()),
        "element_size": tensor.element_size(),
        "numel": tensor.numel(),
        "storage_nbytes": tensor.untyped_storage().nbytes(),
    }


def collect_tensor_storages(
    root: Any,
    *,
    path: str = "root",
    depth: int = 0,
    seen_objects: set[int] | None = None,
    seen_storages: dict[int, int] | None = None,
) -> dict[int, int]:
    if seen_objects is None:
        seen_objects = set()
    if seen_storages is None:
        seen_storages = {}
    if depth > 8:
        return seen_storages
    object_id = id(root)
    if object_id in seen_objects:
        return seen_storages
    seen_objects.add(object_id)

    if torch.is_tensor(root):
        if root.device.type == "cuda":
            storage = root.untyped_storage()
            storage_id = storage.data_ptr()
            seen_storages[storage_id] = storage.nbytes()
        return seen_storages
    if isinstance(root, dict):
        for key, value in list(root.items()):
            collect_tensor_storages(
                value,
                path=f"{path}.{key}",
                depth=depth + 1,
                seen_objects=seen_objects,
                seen_storages=seen_storages,
            )
    elif isinstance(root, (list, tuple, set, frozenset)):
        for index, value in enumerate(root):
            collect_tensor_storages(
                value,
                path=f"{path}[{index}]",
                depth=depth + 1,
                seen_objects=seen_objects,
                seen_storages=seen_storages,
            )
    elif isinstance(root, torch.nn.Module):
        for name, value in root.state_dict().items():
            collect_tensor_storages(
                value,
                path=f"{path}.{name}",
                depth=depth + 1,
                seen_objects=seen_objects,
                seen_storages=seen_storages,
            )
    return seen_storages


def unique_live_bytes(*roots: Any) -> int:
    storages: dict[int, int] = {}
    seen_objects: set[int] = set()
    for root in roots:
        collect_tensor_storages(
            root,
            seen_objects=seen_objects,
            seen_storages=storages,
        )
    return sum(storages.values())


def make_case(workload: dict[str, Any]) -> DenseAttentionCase:
    return DenseAttentionCase(
        name=workload["name"],
        backend="triton",
        forward_mode=ForwardMode.DECODE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        page_size=PAGE_SIZE,
        prefix_lens=tuple(workload["prefix_lens"]),
    )


def loc_fn_for_case(
    case: DenseAttentionCase, max_context_len: int
) -> Any:
    seed = 2026 + len(case.name) + case.num_kv_heads
    return make_loc_fn(
        "shuffled_pages",
        batch_size=case.batch_size,
        seq_lens=case.seq_lens,
        prefix_lens=case.prefix_lens,
        page_size=case.page_size,
        max_context_len=max_context_len,
        seed=seed,
    )


def install_fp8_pool(fixture: Any) -> None:
    old_pool = fixture.runner.token_to_kv_pool
    fp8_pool = MHATokenToKVPool(
        size=old_pool.size,
        page_size=old_pool.page_size,
        dtype=torch.float8_e4m3fn,
        head_num=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    fixture.runner.token_to_kv_pool = fp8_pool
    fixture.runner.token_to_kv_pool_allocator.get_kvcache = lambda: fp8_pool
    fixture.runner.kv_cache_dtype = torch.float8_e4m3fn
    fixture.backend.token_to_kv_pool = fp8_pool
    fixture.actual_module.attn.k_scale = FP8_K_SCALE
    fixture.actual_module.attn.v_scale = FP8_V_SCALE
    fixture.actual_module.attn.k_scale_float = FP8_K_SCALE
    fixture.actual_module.attn.v_scale_float = FP8_V_SCALE


def populate_prefix_cache(fixture: Any, case: DenseAttentionCase) -> None:
    module = fixture.actual_module
    pool = fixture.runner.token_to_kv_pool
    locations = []
    keys = []
    values = []
    for request_index, prefix_hidden in enumerate(fixture.prefix_hidden):
        _, prefix_k, prefix_v = module.project_qkv(prefix_hidden)
        prefix_k = prefix_k.view(-1, case.num_kv_heads, HEAD_DIM)
        prefix_v = prefix_v.view(-1, case.num_kv_heads, HEAD_DIM)
        request_locations = fixture.runner.req_to_token_pool.req_to_token[
            request_index, : case.prefix_lens[request_index]
        ].long()
        locations.append(request_locations)
        keys.append(prefix_k)
        values.append(prefix_v)
    pool.set_kv_buffer(
        module.attn,
        torch.cat(locations, dim=0),
        torch.cat(keys, dim=0),
        torch.cat(values, dim=0),
        FP8_K_SCALE if pool.dtype == torch.float8_e4m3fn else None,
        FP8_V_SCALE if pool.dtype == torch.float8_e4m3fn else None,
    )


def advance_decode_fixture(
    fixture: Any,
    previous_case: DenseAttentionCase,
    max_context_len: int,
    input_hidden: torch.Tensor,
) -> DenseAttentionCase:
    next_case = replace(
        previous_case,
        prefix_lens=tuple(length + 1 for length in previous_case.prefix_lens),
    )
    next_loc_fn = loc_fn_for_case(next_case, max_context_len)
    for request_index, prefix_length in enumerate(next_case.prefix_lens):
        current_position = prefix_length
        fixture.runner.req_to_token_pool.req_to_token[
            request_index, current_position
        ] = next_loc_fn(request_index, current_position)

    seq_lens = next_case.seq_lens
    fixture.case = next_case
    fixture.input_hidden = input_hidden
    fixture.forward_batch.seq_lens = torch.tensor(
        seq_lens, dtype=torch.int32, device="cuda"
    )
    fixture.forward_batch.seq_lens_cpu = torch.tensor(
        seq_lens, dtype=torch.int32, device="cpu"
    )
    fixture.forward_batch.seq_lens_sum = sum(seq_lens)
    fixture.forward_batch.positions = torch.tensor(
        [length - 1 for length in seq_lens], dtype=torch.int64, device="cuda"
    )
    fixture.forward_batch.out_cache_loc = torch.tensor(
        [
            next_loc_fn(request_index, length - 1)
            for request_index, length in enumerate(seq_lens)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    return next_case


def cache_reference(
    fixture: Any,
    case: DenseAttentionCase,
    input_hidden: torch.Tensor,
) -> torch.Tensor:
    module = fixture.actual_module
    pool = fixture.runner.token_to_kv_pool
    k_scale = FP8_K_SCALE if pool.dtype == torch.float8_e4m3fn else 1.0
    v_scale = FP8_V_SCALE if pool.dtype == torch.float8_e4m3fn else 1.0
    query, _, _ = module.project_qkv(input_hidden)
    query = query.view(-1, case.num_heads, HEAD_DIM)
    outputs = []
    for request_index, seq_len in enumerate(case.seq_lens):
        cache_locations = fixture.runner.req_to_token_pool.req_to_token[
            request_index, :seq_len
        ].long()
        keys = pool.get_key_buffer(0)[cache_locations].movedim(0, 1).float()
        values = pool.get_value_buffer(0)[cache_locations].movedim(0, 1).float()
        keys = keys * k_scale
        values = values * v_scale
        if case.num_kv_heads != case.num_heads:
            repeat_factor = case.num_heads // case.num_kv_heads
            keys = keys.repeat_interleave(repeat_factor, dim=0)
            values = values.repeat_interleave(repeat_factor, dim=0)
        scores = torch.einsum(
            "hd,hkd->hk", query[request_index].float(), keys
        ) * module.attn.scaling
        probabilities = torch.softmax(scores, dim=-1)
        attention_output = torch.einsum("hk,hkd->hd", probabilities, values)
        outputs.append(attention_output.reshape(-1))
    return module.o_proj(torch.stack(outputs, dim=0).to(input_hidden.dtype))


def install_dispatch_recorder(fixture: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    original_decode = fixture.backend.decode_attention_fwd

    def recorded_decode(*args: Any, **kwargs: Any) -> Any:
        records.append(
            {
                "q_shape": list(args[0].shape),
                "k_cache_dtype": dtype_name(args[1].dtype),
                "v_cache_dtype": dtype_name(args[2].dtype),
                "output_shape": list(args[3].shape),
                "kv_indptr_shape": list(args[4].shape),
                "kv_indices_shape": list(args[5].shape),
                "attn_logits_shape": list(args[6].shape),
                "attn_lse_shape": list(args[7].shape),
                "num_kv_splits_shape": list(args[8].shape),
                "k_descale": args[11],
                "v_descale": args[12],
                "enable_lean": kwargs.get("enable_lean"),
            }
        )
        return original_decode(*args, **kwargs)

    fixture.backend.decode_attention_fwd = recorded_decode
    return records


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(percentile_value / 100.0 * len(ordered)) - 1),
    )
    return ordered[index]


def profile_one_forward(fixture: Any) -> list[dict[str, Any]]:
    with profile(activities=[ProfilerActivity.CUDA]) as profiler_result:
        run_dense_fixture_eager(fixture)
        torch.cuda.synchronize()
    kernel_stats: dict[str, dict[str, Any]] = {}
    for event in profiler_result.events():
        if not str(event.device_type).endswith("CUDA"):
            continue
        name = event.name
        stats = kernel_stats.setdefault(
            name, {"count": 0, "self_device_time_us": 0.0}
        )
        stats["count"] += 1
        stats["self_device_time_us"] += float(event.self_device_time_total)
    return sorted(
        (
            {
                "name": name,
                **stats,
            }
            for name, stats in kernel_stats.items()
        ),
        key=lambda item: item["self_device_time_us"],
        reverse=True,
    )[:20]


def run_workload(
    workload: dict[str, Any], kv_dtype: torch.dtype
) -> dict[str, Any]:
    case = make_case(workload)
    max_context_len = workload["max_context_len"]
    torch.manual_seed(2026 + len(case.name) + case.num_kv_heads)
    torch.cuda.manual_seed_all(2026 + len(case.name) + case.num_kv_heads)
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    fixture = build_dense_attention_fixture(
        NoopTest(),
        case,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
        max_context_len=max_context_len,
        dtype=torch.bfloat16,
        device="cuda",
        loc_layout="shuffled_pages",
    )
    if kv_dtype == torch.float8_e4m3fn:
        install_fp8_pool(fixture)
    populate_prefix_cache(fixture, case)
    dispatch_records = install_dispatch_recorder(fixture)

    step_results = []
    actual_outputs_cpu = []
    current_case = case
    current_input = fixture.input_hidden
    for step_index in range(RECURRENT_STEPS):
        if step_index > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(910_000 + len(case.name) * 100 + step_index)
            current_input = torch.randn(
                case.batch_size,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cpu",
                generator=generator,
            ).to("cuda")
            current_case = advance_decode_fixture(
                fixture,
                current_case,
                max_context_len,
                current_input,
            )
        dispatch_records.clear()
        actual_output = run_dense_fixture_eager(fixture)
        post_forward_allocated = torch.cuda.memory_allocated()
        reference_output = cache_reference(fixture, current_case, current_input)
        max_abs_error = (
            actual_output.float() - reference_output.float()
        ).abs().max().item()
        finite = bool(
            torch.isfinite(actual_output).all().item()
            and torch.isfinite(reference_output).all().item()
        )
        passed = finite and torch.allclose(
            actual_output, reference_output, atol=NUMERICAL_ATOL, rtol=NUMERICAL_RTOL
        )
        actual_outputs_cpu.append(actual_output.detach().cpu().clone())
        step_results.append(
            {
                "step": step_index,
                "prefix_lens": list(current_case.prefix_lens),
                "seq_lens": list(current_case.seq_lens),
                "input_tokens": current_case.num_input_tokens,
                "max_abs_error_vs_dequantized_cache_reference": max_abs_error,
                "finite": finite,
                "gate_passed": passed,
                "post_forward_live_bytes": post_forward_allocated,
                "dispatch": list(dispatch_records),
            }
        )
        if not passed:
            raise AssertionError(
                f"Numerical gate failed for {workload['name']} {dtype_name(kv_dtype)} "
                f"step {step_index}: max_abs_error={max_abs_error}"
            )
        del actual_output, reference_output

    for _ in range(WARMUP_FORWARDS):
        run_dense_fixture_eager(fixture)
    torch.cuda.synchronize()
    timing_records = []
    dispatch_records.clear()
    for _ in range(TIMED_FORWARDS):
        start_ns = time.perf_counter_ns()
        timed_output = run_dense_fixture_eager(fixture)
        torch.cuda.synchronize()
        end_ns = time.perf_counter_ns()
        timing_records.append((end_ns - start_ns) / 1_000_000.0)
        del timed_output
    timed_outputs_consistent = True
    timing_output_first = run_dense_fixture_eager(fixture)
    timing_output_second = run_dense_fixture_eager(fixture)
    timed_outputs_consistent = torch.equal(timing_output_first, timing_output_second)
    del timing_output_first, timing_output_second

    timed_dispatch_count = len(dispatch_records)
    dispatch_records.clear()
    kernel_events = profile_one_forward(fixture)
    profiler_dispatch_count = len(dispatch_records)
    final_allocated = torch.cuda.memory_allocated()
    final_reserved = torch.cuda.memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    if max(baseline_allocated, final_allocated, peak_allocated) >= MAX_LIVE_BYTES:
        raise RuntimeError(
            "Live allocation limit exceeded: "
            f"baseline={baseline_allocated}, final={final_allocated}, peak={peak_allocated}"
        )

    pool = fixture.runner.token_to_kv_pool
    module = fixture.actual_module
    kv_cache_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*pool.k_buffer, *pool.v_buffer)
    )
    model_weight_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )
    req_to_token_bytes = (
        fixture.runner.req_to_token_pool.req_to_token.numel()
        * fixture.runner.req_to_token_pool.req_to_token.element_size()
    )
    backend_tensor_bytes = unique_live_bytes(
        fixture.backend.__dict__, fixture.backend.forward_metadata.__dict__
    )
    forward_batch_bytes = unique_live_bytes(fixture.forward_batch.__dict__)
    input_output_bytes = (
        fixture.input_hidden.numel() * fixture.input_hidden.element_size()
    )
    predicted_live_bytes = unique_live_bytes(
        module,
        fixture.runner.req_to_token_pool.__dict__,
        pool.__dict__,
        fixture.backend.__dict__,
        fixture.backend.forward_metadata.__dict__,
        fixture.forward_batch.__dict__,
        fixture.input_hidden,
    )
    cache_slots = pool.k_buffer[0].shape[0]
    result = {
        "name": workload["name"],
        "kv_dtype": dtype_name(kv_dtype),
        "model_dtype": "bfloat16",
        "backend": "triton",
        "forward_mode": "DECODE",
        "batch_size": case.batch_size,
        "initial_prefix_lens": list(case.prefix_lens),
        "final_prefix_lens": list(current_case.prefix_lens),
        "final_seq_lens": list(current_case.seq_lens),
        "num_heads": NUM_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN_SIZE,
        "page_size": PAGE_SIZE,
        "max_context_len": max_context_len,
        "kv_scales": {
            "k": FP8_K_SCALE if kv_dtype == torch.float8_e4m3fn else 1.0,
            "v": FP8_V_SCALE if kv_dtype == torch.float8_e4m3fn else 1.0,
        },
        "recurrent_steps": step_results,
        "timing": {
            "warmup_forwards": WARMUP_FORWARDS,
            "timed_forwards": TIMED_FORWARDS,
            "method": "time.perf_counter_ns around full eager forward plus torch.cuda.synchronize",
            "latency_ms": timing_records,
            "median_ms": statistics.median(timing_records),
            "p90_ms": percentile(timing_records, 90),
            "p99_ms": percentile(timing_records, 99),
            "min_ms": min(timing_records),
            "max_ms": max(timing_records),
            "timed_outputs_identical": timed_outputs_consistent,
        },
        "dispatch": {
            "backend_method": "TritonAttnBackend.forward_decode",
            "kernel_wrapper": "decode_attention_fwd",
            "timed_dispatch_count": timed_dispatch_count,
            "profiler_kernel_events": kernel_events,
            "profiler_dispatch_count": profiler_dispatch_count,
        },
        "allocation": {
            "baseline_live_bytes": baseline_allocated,
            "final_live_bytes": final_allocated,
            "peak_live_bytes": peak_allocated,
            "reserved_bytes": final_reserved,
            "predicted_kv_cache_bytes": kv_cache_bytes,
            "predicted_model_weight_bytes": model_weight_bytes,
            "predicted_req_to_token_bytes": req_to_token_bytes,
            "predicted_backend_tensor_bytes": backend_tensor_bytes,
            "predicted_forward_batch_bytes": forward_batch_bytes,
            "predicted_input_bytes": input_output_bytes,
            "predicted_live_bytes": predicted_live_bytes,
            "measured_minus_predicted_live_bytes": final_allocated
            - predicted_live_bytes,
            "measured_live_delta_from_baseline": final_allocated
            - baseline_allocated,
            "measured_peak_delta_from_baseline": peak_allocated
            - baseline_allocated,
            "cache_slots": cache_slots,
            "kv_bytes_per_slot": kv_cache_bytes / cache_slots,
            "k_cache": tensor_summary(pool.k_buffer[0]),
            "v_cache": tensor_summary(pool.v_buffer[0]),
            "pool_dtype": dtype_name(pool.dtype),
            "pool_store_dtype": dtype_name(pool.store_dtype),
        },
        "actual_outputs_cpu": [tensor.tolist() for tensor in actual_outputs_cpu],
    }
    return result


def software_record() -> dict[str, Any]:
    import sglang
    import triton

    device_properties = torch.cuda.get_device_properties(0)
    return {
        "python": os.sys.executable,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "sglang": getattr(sglang, "__version__", None),
        "sglang_path": sglang.__file__,
        "triton_path": triton.__file__,
        "gpu_name": device_properties.name,
        "gpu_total_memory_bytes": device_properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": "bounded MI300X Triton MHA BF16/FP8 KV decode benchmark",
        "software": software_record(),
        "limits": {
            "workload_cases": len(WORKLOADS) * 2,
            "recurrent_steps_per_case": RECURRENT_STEPS,
            "warmup_forwards_per_case": WARMUP_FORWARDS,
            "timed_forwards_per_case": TIMED_FORWARDS,
            "max_live_bytes": MAX_LIVE_BYTES,
            "model_weights_target_bytes": 4 * 1024**3,
        },
        "numerical_gate": {
            "atol": NUMERICAL_ATOL,
            "rtol": NUMERICAL_RTOL,
            "reference": "independent attention over cache rows; FP8 rows are dequantized with explicit scales",
        },
        "workloads": [],
    }
    outputs_by_workload: dict[str, dict[str, list[list[float]]]] = {}
    for workload in WORKLOADS:
        outputs_by_workload[workload["name"]] = {}
        for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            result = run_workload(workload, kv_dtype)
            results["workloads"].append(result)
            outputs_by_workload[workload["name"]][dtype_name(kv_dtype)] = result[
                "actual_outputs_cpu"
            ]
            del result["actual_outputs_cpu"]
            torch.cuda.empty_cache()

    cross_dtype = []
    for workload in WORKLOADS:
        bf16_outputs = outputs_by_workload[workload["name"]]["bfloat16"]
        fp8_outputs = outputs_by_workload[workload["name"]]["float8_e4m3fn"]
        step_errors = []
        for bf16_output, fp8_output in zip(bf16_outputs, fp8_outputs):
            step_errors.append(
                (
                    torch.tensor(bf16_output).float()
                    - torch.tensor(fp8_output).float()
                )
                .abs()
                .max()
                .item()
            )
        cross_dtype.append(
            {
                "name": workload["name"],
                "max_abs_error_by_step": step_errors,
                "max_abs_error": max(step_errors),
            }
        )
    results["cross_dtype_output_comparison"] = cross_dtype
    results["notes"] = [
        "All model weights are locally generated by the fixture; no checkpoint is downloaded.",
        "BF16 uses unit scales. FP8 uses k_scale=0.03125 and v_scale=0.0625.",
        "Timing includes metadata initialization, projections, cache write, and attention forward.",
        "Predicted live bytes counts unique CUDA tensor storages reachable from the fixture; the delta includes global/import allocations.",
        "This is a numerical and allocation experiment, not a model-quality claim.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
