#!/usr/bin/env python3
"""Bounded BF16/FP8-KV dummy-model decode benchmark on one HIP GPU.

The benchmark uses the real ProjectedDenseAttention block (Q/K/V/O projections
plus RadixAttention) and the Aiter backend. It does not load model weights and
does not make model-quality claims.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, scaled_fp8_quant
from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_schedule
from sglang.test.kits.attention_unittest.attention_methods import dense_attention as kit


PAGE_SIZE = 16
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 128
HIDDEN_SIZE = 512
DECODE_STEPS = 4
WARMUP_SEQUENCES = 3
MEASURED_SEQUENCES = 15
K_SCALE = 0.02
V_SCALE = 0.03
MAX_WEIGHT_BYTES = 4 * 1024**3
MAX_LIVE_ALLOC_BYTES = 48 * 1024**3
MAX_WORKLOAD_CASES = 6
WORKLOAD_CASES = (
    (1, 128),
    (1, 512),
    (2, 128),
    (2, 512),
    (4, 128),
    (4, 512),
)
ACCURACY_GATES = {
    "bf16": {"atol": 0.05, "rtol": 0.05, "min_cosine": 0.999},
    "fp8": {
        "atol": 0.15,
        "rtol": 0.15,
        "min_cosine": 0.99,
        "max_mismatch_fraction": 0.005,
    },
}


@dataclass(frozen=True)
class CaseSpec:
    batch_size: int
    prefix_len: int


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def tensor_bytes(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def make_case(spec: CaseSpec, prefix_len: int) -> kit.DenseAttentionCase:
    return kit.DenseAttentionCase(
        name=f"b{spec.batch_size}_p{spec.prefix_len}_s{prefix_len}",
        backend="aiter",
        forward_mode=ForwardMode.DECODE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        page_size=PAGE_SIZE,
        prefix_lens=(prefix_len,) * spec.batch_size,
    )


def build_run(spec: CaseSpec, kv_dtype: torch.dtype) -> dict[str, Any]:
    max_context_len = ((spec.prefix_len + DECODE_STEPS + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    initial_case = make_case(spec, spec.prefix_len)
    model_config = kit.TinyModelConfig(
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
        context_len=max_context_len,
    )
    model_config.dtype = torch.bfloat16
    runner = kit.MockModelRunner(
        case=initial_case,
        model_config=model_config,
        dtype=torch.bfloat16,
        device="cuda",
        max_context_len=max_context_len,
        head_dim=HEAD_DIM,
    )
    runner.kv_cache_dtype = kv_dtype
    runner.kv_cache_dtype_str = "auto" if kv_dtype == torch.bfloat16 else "fp8_e4m3"
    pool_size = PAGE_SIZE + spec.batch_size * max_context_len + PAGE_SIZE
    runner.token_to_kv_pool = MHATokenToKVPool(
        size=pool_size,
        page_size=PAGE_SIZE,
        dtype=kv_dtype,
        head_num=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
    )

    with get_schedule().override(page_size=PAGE_SIZE):
        backend = ATTENTION_BACKENDS["aiter"](runner)
        module = kit.ProjectedDenseAttention(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        reference_module = kit.ReferenceDenseAttention(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        kit._copy_dense_weights(module, reference_module)

        if kv_dtype == fp8_dtype:
            module.attn.k_scale = torch.tensor([K_SCALE], device="cuda")
            module.attn.v_scale = torch.tensor([V_SCALE], device="cuda")

        seed = 61783 + spec.batch_size * 1009 + spec.prefix_len
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        prefix_hidden = [
            torch.randn(
                spec.prefix_len,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            for _ in range(spec.batch_size)
        ]
        step_hidden = [
            torch.randn(
                spec.batch_size,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            for _ in range(DECODE_STEPS)
        ]

        keys = []
        values = []
        locations = []
        for request_index, hidden in enumerate(prefix_hidden):
            _, key, value = module.project_qkv(hidden)
            keys.append(key.view(-1, NUM_KV_HEADS, HEAD_DIM))
            values.append(value.view(-1, NUM_KV_HEADS, HEAD_DIM))
            locations.extend(
                kit._token_loc(
                    request_index,
                    position,
                    page_size=PAGE_SIZE,
                    max_context_len=max_context_len,
                )
                for position in range(spec.prefix_len)
            )
        location_tensor = torch.tensor(locations, dtype=torch.int64, device="cuda")
        if kv_dtype == fp8_dtype:
            runner.token_to_kv_pool.set_kv_buffer(
                module.attn,
                location_tensor,
                torch.cat(keys),
                torch.cat(values),
                module.attn.k_scale,
                module.attn.v_scale,
            )
        else:
            runner.token_to_kv_pool.set_kv_buffer(
                module.attn,
                location_tensor,
                torch.cat(keys),
                torch.cat(values),
            )

    return {
        "spec": spec,
        "max_context_len": max_context_len,
        "runner": runner,
        "backend": backend,
        "module": module,
        "reference_module": reference_module,
        "prefix_hidden": prefix_hidden,
        "step_hidden": step_hidden,
    }


def make_batch(run: dict[str, Any], step_index: int):
    spec = run["spec"]
    prefix_len = spec.prefix_len + step_index
    case = make_case(spec, prefix_len)
    return kit._make_forward_batch(
        case,
        run["runner"],
        max_context_len=run["max_context_len"],
        device="cuda",
    )


def forward_once(run: dict[str, Any], step_index: int) -> torch.Tensor:
    batch = make_batch(run, step_index)
    with torch.no_grad(), forward_context(
        ForwardContext(attn_backend=run["backend"])
    ):
        run["backend"].init_forward_metadata(batch)
        return run["module"](run["step_hidden"][step_index], batch)


def dequantized_reference(run: dict[str, Any], step_index: int) -> torch.Tensor:
    spec = run["spec"]
    sequence_len = spec.prefix_len + step_index + 1
    module = run["reference_module"]
    hidden = run["step_hidden"][step_index]
    query = module.q_proj(hidden).view(spec.batch_size, NUM_HEADS, HEAD_DIM)
    key_cache, value_cache = run["runner"].token_to_kv_pool.get_kv_buffer(0)
    locations = run["runner"].req_to_token_pool.req_to_token[
        : spec.batch_size, :sequence_len
    ].long()

    if run["runner"].kv_cache_dtype == fp8_dtype:
        query_fp8, _ = scaled_fp8_quant(
            query.reshape(spec.batch_size, -1),
            torch.tensor([K_SCALE], device="cuda"),
        )
        query_ref = (
            query_fp8.view(spec.batch_size, NUM_HEADS, HEAD_DIM).float() * K_SCALE
        )
        keys = key_cache[locations].float() * K_SCALE
        values = value_cache[locations].float() * V_SCALE
    else:
        query_ref = query.float()
        keys = key_cache[locations].float()
        values = value_cache[locations].float()

    if NUM_KV_HEADS != NUM_HEADS:
        repeat = NUM_HEADS // NUM_KV_HEADS
        keys = keys.repeat_interleave(repeat, dim=2)
        values = values.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bhd,bthd->bht", query_ref, keys) / math.sqrt(HEAD_DIM)
    probabilities = torch.softmax(scores, dim=-1)
    attention = torch.einsum("bht,bthd->bhd", probabilities, values)
    return module.o_proj(
        attention.to(torch.bfloat16).reshape(spec.batch_size, NUM_HEADS * HEAD_DIM)
    )


def accuracy(actual: torch.Tensor, expected: torch.Tensor, kv_dtype: torch.dtype):
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    gate = ACCURACY_GATES["fp8" if kv_dtype == fp8_dtype else "bf16"]
    tolerance = gate["atol"] + gate["rtol"] * expected_float.abs()
    mismatch_fraction = (difference > tolerance).float().mean().item()
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "cosine": cosine,
        "mismatch_fraction": mismatch_fraction,
        "all_finite": bool(torch.isfinite(actual_float).all()),
    }


def assert_accuracy(metrics: dict[str, Any], kv_dtype: torch.dtype) -> None:
    gate = ACCURACY_GATES["fp8" if kv_dtype == fp8_dtype else "bf16"]
    assert metrics["all_finite"], "non-finite block output"
    assert metrics["cosine"] >= gate["min_cosine"], metrics
    if "max_mismatch_fraction" in gate:
        assert (
            metrics["mismatch_fraction"] <= gate["max_mismatch_fraction"]
        ), metrics
    else:
        assert metrics["max_abs_error"] <= gate["atol"], metrics


def correctness_pass(run: dict[str, Any]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    expected_outputs = []
    accuracy_metrics = []
    for step_index in range(DECODE_STEPS):
        actual = forward_once(run, step_index)
        expected = dequantized_reference(run, step_index)
        metrics = accuracy(actual, expected, run["runner"].kv_cache_dtype)
        assert_accuracy(metrics, run["runner"].kv_cache_dtype)
        expected_outputs.append(expected.detach().clone())
        accuracy_metrics.append(metrics)
    return expected_outputs, accuracy_metrics


def timed_sequence(
    run: dict[str, Any], expected_outputs: list[torch.Tensor]
) -> tuple[list[float], float]:
    step_milliseconds = []
    step_events = []
    sequence_start = torch.cuda.Event(enable_timing=True)
    sequence_end = torch.cuda.Event(enable_timing=True)
    sequence_start.record()
    outputs = []
    for step_index in range(DECODE_STEPS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        batch = make_batch(run, step_index)
        with torch.no_grad(), forward_context(
            ForwardContext(attn_backend=run["backend"])
        ):
            run["backend"].init_forward_metadata(batch)
            start.record()
            output = run["module"](run["step_hidden"][step_index], batch)
            end.record()
        outputs.append(output)
        step_events.append((start, end))
    sequence_end.record()
    torch.cuda.synchronize()
    for step_index, output in enumerate(outputs):
        metrics = accuracy(
            output, expected_outputs[step_index], run["runner"].kv_cache_dtype
        )
        assert_accuracy(metrics, run["runner"].kv_cache_dtype)
    for start, end in step_events:
        step_milliseconds.append(start.elapsed_time(end))
    return step_milliseconds, sequence_start.elapsed_time(sequence_end)


def record_dispatch(run: dict[str, Any]) -> dict[str, int]:
    original_unified = aiter_backend.unified_attention
    original_paged = aiter_backend.paged_attention_ragged
    original_write = aiter_backend.launch_reshape_and_cache_flash
    counts = {"unified_attention": 0, "paged_attention_ragged": 0, "fused_fp8_write": 0}

    def unified_wrapper(*args, **kwargs):
        counts["unified_attention"] += 1
        return original_unified(*args, **kwargs)

    def paged_wrapper(*args, **kwargs):
        counts["paged_attention_ragged"] += 1
        return original_paged(*args, **kwargs)

    def write_wrapper(*args, **kwargs):
        counts["fused_fp8_write"] += 1
        return original_write(*args, **kwargs)

    aiter_backend.unified_attention = unified_wrapper
    aiter_backend.paged_attention_ragged = paged_wrapper
    aiter_backend.launch_reshape_and_cache_flash = write_wrapper
    try:
        correctness_pass(run)
    finally:
        aiter_backend.unified_attention = original_unified
        aiter_backend.paged_attention_ragged = original_paged
        aiter_backend.launch_reshape_and_cache_flash = original_write
    return counts


def allocation_record(run: dict[str, Any]) -> dict[str, Any]:
    pool = run["runner"].token_to_kv_pool
    key_cache, value_cache = pool.get_kv_buffer(0)
    backend = run["backend"]
    module_tensors = [tensor for tensor in run["module"].parameters()]
    backend_tensors = [
        tensor
        for tensor in (
            getattr(backend, "kv_indptr", None),
            getattr(backend, "kv_last_page_len", None),
            getattr(backend, "qo_indptr", None),
            getattr(backend, "qo_indptr_unified_decode", None),
            getattr(backend, "mask_indptr", None),
            getattr(backend, "workspace_buffer", None),
        )
        if isinstance(tensor, torch.Tensor)
    ]
    return {
        "pool_size_slots": pool.size,
        "pool_padding_slots": pool.page_size,
        "page_size": pool.page_size,
        "key_cache_shape": list(key_cache.shape),
        "value_cache_shape": list(value_cache.shape),
        "key_cache_dtype": str(key_cache.dtype),
        "value_cache_dtype": str(value_cache.dtype),
        "key_cache_bytes": key_cache.numel() * key_cache.element_size(),
        "value_cache_bytes": value_cache.numel() * value_cache.element_size(),
        "req_to_token_shape": list(
            run["runner"].req_to_token_pool.req_to_token.shape
        ),
        "req_to_token_bytes": run["runner"].req_to_token_pool.req_to_token.numel()
        * run["runner"].req_to_token_pool.req_to_token.element_size(),
        "backend_tensor_bytes": tensor_bytes(backend_tensors),
        "locally_generated_weight_bytes": tensor_bytes(module_tensors),
        "dispatch": {
            "backend": "AiterAttnBackend",
            "use_triton_unified_attention": bool(
                backend.use_triton_unified_attention
            ),
            "kv_cache_is_vectorized_5d": bool(backend.kv_cache_is_vectorized_5d),
            "kv_cache_dtype": str(backend.kv_cache_dtype),
            "input_dtype": str(backend.input_dtype),
        },
    }


def run_case(spec: CaseSpec, kv_dtype: torch.dtype) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run = build_run(spec, kv_dtype)
    weight_bytes = tensor_bytes(list(run["module"].parameters()))
    assert weight_bytes < MAX_WEIGHT_BYTES, weight_bytes
    dispatch_counts = record_dispatch(run)
    expected_outputs, accuracy_metrics = correctness_pass(run)
    for _ in range(WARMUP_SEQUENCES):
        outputs = [forward_once(run, step) for step in range(DECODE_STEPS)]
        for step, output in enumerate(outputs):
            assert_accuracy(
                accuracy(output, expected_outputs[step], kv_dtype), kv_dtype
            )

    step_latencies = []
    sequence_latencies = []
    for _ in range(MEASURED_SEQUENCES):
        step_latency, sequence_latency = timed_sequence(run, expected_outputs)
        step_latencies.extend(step_latency)
        sequence_latencies.append(sequence_latency)

    allocation = allocation_record(run)
    allocation["dispatch"]["observed_calls_per_4_step_correctness_pass"] = dispatch_counts
    peak_allocated = torch.cuda.max_memory_allocated()
    assert peak_allocated < MAX_LIVE_ALLOC_BYTES, peak_allocated
    result = {
        "case": asdict(spec),
        "kv_dtype": str(kv_dtype),
        "query_dtype": "torch.bfloat16",
        "block_output_dtype": "torch.bfloat16",
        "k_scale": K_SCALE if kv_dtype == fp8_dtype else None,
        "v_scale": V_SCALE if kv_dtype == fp8_dtype else None,
        "accuracy_gates": ACCURACY_GATES[
            "fp8" if kv_dtype == fp8_dtype else "bf16"
        ],
        "accuracy": accuracy_metrics,
        "allocation": allocation,
        "peak_allocated_bytes": peak_allocated,
        "timing": {
            "step_median_ms": statistics.median(step_latencies),
            "step_p95_ms": percentile(step_latencies, 0.95),
            "step_max_ms": max(step_latencies),
            "sequence_median_ms": statistics.median(sequence_latencies),
            "sequence_p95_ms": percentile(sequence_latencies, 0.95),
            "step_samples": len(step_latencies),
            "sequence_samples": len(sequence_latencies),
            "raw_step_latencies_ms": step_latencies,
            "raw_sequence_latencies_ms": sequence_latencies,
        },
    }
    del run
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "one CUDA/HIP GPU is required"
    assert torch.cuda.device_count() == 1, "exactly one assigned GPU is required"
    assert len(WORKLOAD_CASES) <= MAX_WORKLOAD_CASES
    torch.cuda.set_device(0)

    results = []
    for batch_size, prefix_len in WORKLOAD_CASES:
        spec = CaseSpec(batch_size=batch_size, prefix_len=prefix_len)
        for kv_dtype in (torch.bfloat16, fp8_dtype):
            results.append(run_case(spec, kv_dtype))

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "decode-sized dummy-model reduced-block latency study; no model-quality claim",
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "gcn_arch_name": torch.cuda.get_device_properties(0).gcnArchName,
            "uuid": str(torch.cuda.get_device_properties(0).uuid),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "multi_processor_count": torch.cuda.get_device_properties(
                0
            ).multi_processor_count,
            "device_count": torch.cuda.device_count(),
        },
        "environment": {
            "python": sys.executable,
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "sglang_test_kit_path": kit.__file__,
            "aiter_backend_path": aiter_backend.__file__,
            "aiter_root_dir": os.environ.get("AITER_ROOT_DIR"),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        },
        "source_commit": git_commit,
        "dimensions": {
            "page_size": PAGE_SIZE,
            "num_query_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "hidden_size": HIDDEN_SIZE,
            "decode_steps": DECODE_STEPS,
            "workload_cases": [asdict(CaseSpec(*case)) for case in WORKLOAD_CASES],
        },
        "dtypes": {
            "query": "torch.bfloat16",
            "block_output": "torch.bfloat16",
            "bf16_kv": "torch.bfloat16",
            "fp8_kv": str(fp8_dtype),
        },
        "scales": {"k": K_SCALE, "v": V_SCALE, "query_uses_k_scale": True},
        "timing_method": {
            "warmup_sequences": WARMUP_SEQUENCES,
            "measured_sequences": MEASURED_SEQUENCES,
            "steps_per_sequence": DECODE_STEPS,
            "method": "CUDA events around each real ProjectedDenseAttention forward; metadata setup excluded",
            "sequence_method": "CUDA events around the full four-step sequence; includes metadata construction/init and Python dispatch between forwards",
            "tail": "p95 by linear interpolation over 60 step samples and 15 sequence samples",
        },
        "limits": {
            "wall_minutes": 120,
            "max_workload_cases": MAX_WORKLOAD_CASES,
            "max_locally_generated_weight_bytes": MAX_WEIGHT_BYTES,
            "max_live_alloc_bytes": MAX_LIVE_ALLOC_BYTES,
            "model_weights_downloaded": False,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
