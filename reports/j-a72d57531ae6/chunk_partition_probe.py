import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLA_ATOL,
    MLA_RTOL,
    MLAAttentionCase,
    _copy_mla_weights,
    _populate_prefix_kv,
    _token_loc,
    build_mla_attention_fixture,
    expected_mla_fixture_output,
    run_mla_fixture_eager,
)


BACKEND = "triton"
DTYPE = torch.float16
DEVICE = "cuda"
HIDDEN_SIZE = 64
KV_LORA_RANK = 32
QK_ROPE_HEAD_DIM = 0
NUM_HEADS = 4
PAGE_SIZE = 1
WARM_REPEATS = 3
SEED = 20260910


def _sync() -> None:
    torch.cuda.synchronize()


def _timed_forward(fixture) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    output = run_mla_fixture_eager(fixture)
    _sync()
    return output, time.perf_counter() - start


def _build_case(name: str, prefix_len: int, extend_len: int, max_context_len: int):
    return MLAAttentionCase(
        name=name,
        backend=BACKEND,
        forward_mode=ForwardMode.EXTEND,
        num_heads=NUM_HEADS,
        page_size=PAGE_SIZE,
        prefix_lens=(prefix_len,),
        extend_lens=(extend_len,),
    ), max_context_len


def _load_weights(fixture, state_dict) -> None:
    fixture.actual_module.load_state_dict(state_dict)
    _copy_mla_weights(fixture.actual_module, fixture.reference_module)


def _cache_rows(fixture, locations: list[int]) -> torch.Tensor:
    pool = fixture.runner.token_to_kv_pool
    index = torch.tensor(locations, dtype=torch.int64, device=fixture.runner.device)
    return pool.kv_buffer[0][index].detach().clone()


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "passes_unchanged_gate": bool(
            torch.allclose(actual, expected, atol=MLA_ATOL, rtol=MLA_RTOL)
        ),
    }


def _run_partition(
    total_tokens: int,
    partitions: tuple[int, ...],
    common_hidden: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    max_context_len: int,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    locations = [
        _token_loc(0, position, page_size=PAGE_SIZE, max_context_len=max_context_len)
        for position in range(total_tokens)
    ]

    unchunked_case, _ = _build_case(
        f"unchunked_{total_tokens}", 0, total_tokens, max_context_len
    )
    unchunked_fixture = build_mla_attention_fixture(
        None,
        unchunked_case,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
        max_context_len=max_context_len,
        dtype=DTYPE,
        device=DEVICE,
        loc_layout="contiguous",
    )
    _load_weights(unchunked_fixture, state_dict)
    unchunked_fixture.prefix_hidden = [common_hidden[:0]]
    unchunked_fixture.input_hidden = common_hidden

    unchunked_cold, unchunked_cold_seconds = _timed_forward(unchunked_fixture)
    unchunked_expected = expected_mla_fixture_output(unchunked_fixture)
    unchunked_reference_error = _error(unchunked_cold, unchunked_expected)

    unchunked_warm = []
    for _ in range(WARM_REPEATS):
        _, elapsed = _timed_forward(unchunked_fixture)
        unchunked_warm.append(elapsed)
    unchunked_cache = _cache_rows(unchunked_fixture, locations)

    chunk_outputs = []
    final_chunk_fixture = None
    chunk_reference_errors = []
    chunk_cold_seconds = []
    chunk_warm_seconds = []
    chunk_memory_bytes = []
    offset = 0

    for chunk_index, chunk_size in enumerate(partitions):
        chunk_case, _ = _build_case(
            f"chunk_{chunk_index}_{chunk_size}", offset, chunk_size, max_context_len
        )
        chunk_fixture = build_mla_attention_fixture(
            None,
            chunk_case,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            hidden_size=HIDDEN_SIZE,
            max_context_len=max_context_len,
            dtype=DTYPE,
            device=DEVICE,
            loc_layout="contiguous",
        )
        _load_weights(chunk_fixture, state_dict)
        chunk_fixture.prefix_hidden = [common_hidden[:offset]]
        chunk_fixture.input_hidden = common_hidden[offset : offset + chunk_size]
        if offset:
            _populate_prefix_kv(
                chunk_fixture.actual_module,
                chunk_case,
                chunk_fixture.runner,
                chunk_fixture.backend,
                chunk_fixture.prefix_hidden,
                max_context_len=max_context_len,
            )

        chunk_cold, cold_elapsed = _timed_forward(chunk_fixture)
        chunk_expected = expected_mla_fixture_output(chunk_fixture)
        chunk_reference_errors.append(_error(chunk_cold, chunk_expected))
        chunk_cold_seconds.append(cold_elapsed)

        warm = []
        for _ in range(WARM_REPEATS):
            _, elapsed = _timed_forward(chunk_fixture)
            warm.append(elapsed)
        chunk_warm_seconds.append(statistics.median(warm))
        chunk_outputs.append(chunk_cold)
        final_chunk_fixture = chunk_fixture
        chunk_memory_bytes.append(torch.cuda.max_memory_allocated())
        offset += chunk_size

    final_chunk_start = total_tokens - partitions[-1]
    final_chunk_output = chunk_outputs[-1]
    unchunked_final_slice = unchunked_cold[final_chunk_start:]
    final_output_error = _error(final_chunk_output, unchunked_final_slice)

    assert final_chunk_fixture is not None
    final_chunk_cache = _cache_rows(final_chunk_fixture, locations)
    final_cache_error = _error(final_chunk_cache, unchunked_cache)

    unchunked_warm_seconds = statistics.median(unchunked_warm)
    chunk_total_warm_seconds = sum(chunk_warm_seconds)
    peak_memory_bytes = max(chunk_memory_bytes)

    return {
        "total_tokens": total_tokens,
        "partitions": list(partitions),
        "unchunked": {
            "cold_seconds": unchunked_cold_seconds,
            "warm_median_seconds": unchunked_warm_seconds,
            "throughput_tokens_per_second": total_tokens / unchunked_warm_seconds,
            "reference_error": unchunked_reference_error,
        },
        "chunked": {
            "cold_seconds_by_chunk": chunk_cold_seconds,
            "warm_median_seconds_by_chunk": chunk_warm_seconds,
            "total_warm_seconds": chunk_total_warm_seconds,
            "throughput_tokens_per_second": total_tokens / chunk_total_warm_seconds,
            "reference_errors_by_chunk": chunk_reference_errors,
            "final_output_vs_unchunked_error": final_output_error,
            "final_cache_vs_unchunked_error": final_cache_error,
        },
        "total_useful_work_tokens": total_tokens,
        "total_useful_work_token_forwards": total_tokens,
        "peak_memory_bytes": peak_memory_bytes,
    }


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    total_lengths = (64, 128, 256, 512, 768, 1024)
    partition_shapes = (
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (256, 256, 256),
        (512, 512),
    )

    first_case, max_context_len = _build_case("weight_seed", 0, 1, 1024)
    weight_fixture = build_mla_attention_fixture(
        None,
        first_case,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
        max_context_len=max_context_len,
        dtype=DTYPE,
        device=DEVICE,
        loc_layout="contiguous",
    )
    state_dict = {
        key: value.detach().clone()
        for key, value in weight_fixture.actual_module.state_dict().items()
    }
    weight_bytes = sum(
        value.numel() * value.element_size() for value in state_dict.values()
    )

    results = []
    for total_tokens, partitions in zip(total_lengths, partition_shapes):
        common_hidden = torch.randn(
            total_tokens,
            HIDDEN_SIZE,
            dtype=DTYPE,
            device=DEVICE,
        )
        result = _run_partition(
            total_tokens,
            partitions,
            common_hidden,
            state_dict,
            max_context_len=1024,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    report = {
        "label": "MI300X reduced DeepSeek MLA chunk-partition probe",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "count": torch.cuda.device_count(),
        },
        "torch": {
            "version": torch.__version__,
            "path": torch.__file__,
            "hip": torch.version.hip,
        },
        "python": os.__file__,
        "native_paths": {
            "torch_c": torch._C.__file__,
            "triton": __import__("triton").__file__,
            "sglang": __import__("sglang").__file__,
        },
        "backend": BACKEND,
        "dtype": str(DTYPE),
        "hidden_size": HIDDEN_SIZE,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "num_heads": NUM_HEADS,
        "page_size": PAGE_SIZE,
        "max_context_len": 1024,
        "seed": SEED,
        "weight_bytes": weight_bytes,
        "unchanged_numerical_gate": {
            "atol": MLA_ATOL,
            "rtol": MLA_RTOL,
        },
        "timing_method": (
            "torch.cuda.synchronize around each forward; cold is first execution, "
            f"warm is median of {WARM_REPEATS} executions"
        ),
        "cases": results,
    }

    output_path = Path(__file__).with_name("results.json")
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
