#!/usr/bin/env python3

import json
import platform
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_dcp_sharded_triton,
    set_mla_kv_scale_buffer_triton,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
)
from sglang.kernels.ops.kvcache.set_mla_kv_buffer import can_use_set_mla_kv_buffer
from sglang.srt.runtime_context import get_parallel


DEVICE = "cuda"
CACHE_SIZE = 64
NOPE_DIM = 128
ROPE_DIM = 64
SCALE_NOPE_DIM = 16
SCALE_ROPE_DIM = 4


def _host_case(loc, nope_dim, rope_dim, dtype, loc_dtype, fp8_dtype=None):
    generator = torch.Generator(device="cpu").manual_seed(36207)
    source_dtype = torch.bfloat16 if dtype == torch.uint8 else dtype
    loc_host = torch.tensor(loc, dtype=loc_dtype)
    batch_size = loc_host.numel()
    cache_host = torch.randn(
        (CACHE_SIZE, 1, nope_dim + rope_dim),
        dtype=torch.float32,
        generator=generator,
    ).to(dtype)
    nope_host = torch.randn(
        (batch_size, 1, nope_dim), dtype=torch.float32, generator=generator
    ).to(source_dtype)
    rope_host = torch.randn(
        (batch_size, 1, rope_dim), dtype=torch.float32, generator=generator
    ).to(source_dtype)

    padding_rows = [index for index, slot in enumerate(loc) if slot == 0]
    nope_host[padding_rows] = float("nan")
    rope_host[padding_rows] = float("nan")

    expected_host = cache_host.clone()
    for row, slot in enumerate(loc_host.tolist()):
        if slot == 0:
            continue
        if fp8_dtype is None:
            expected_host[slot, 0, :nope_dim] = nope_host[row, 0]
            expected_host[slot, 0, nope_dim:] = rope_host[row, 0]
        else:
            packed = torch.cat((nope_host[row, 0], rope_host[row, 0])).to(fp8_dtype)
            expected_host[slot, 0] = packed.view(torch.uint8)

    return (
        cache_host.to(DEVICE),
        loc_host.to(DEVICE),
        nope_host.to(DEVICE),
        rope_host.to(DEVICE),
        cache_host,
        expected_host,
    )


def _run_eager(case_name, writer, cache, loc, nope, rope, expected, fp8_dtype=None):
    if fp8_dtype is None:
        writer(cache, loc, nope, rope)
    else:
        writer(cache, loc, nope, rope, fp8_dtype)
    torch.cuda.synchronize()
    return _check(case_name, cache, expected)


def _run_graph(
    case_name, writer, cache, loc, nope, rope, initial_cache, expected, fp8_dtype=None
):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if fp8_dtype is None:
            writer(cache, loc, nope, rope)
        else:
            writer(cache, loc, nope, rope, fp8_dtype)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if fp8_dtype is None:
            writer(cache, loc, nope, rope)
        else:
            writer(cache, loc, nope, rope, fp8_dtype)

    cache.copy_(initial_cache.to(DEVICE))
    graph.replay()
    torch.cuda.synchronize()
    return _check(case_name, cache, expected)


def _check(case_name, cache, expected):
    actual = cache.detach().cpu()
    expected_cpu = expected.detach().cpu()
    return {
        "case": case_name,
        "slot_zero_unchanged": bool(torch.equal(actual[0], expected_cpu[0])),
        "cache_exactly_matches_reference": bool(torch.equal(actual, expected_cpu)),
        "mismatch_count": int((actual != expected_cpu).sum().item()),
    }


def main():
    torch.cuda.set_device(0)
    loc_dtypes = (torch.int32, torch.int64)
    padded_loc = [0, 0, 2, 3, 4, 5]
    fp8_dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
    results = []

    for loc_dtype in loc_dtypes:
        dtype_name = str(loc_dtype).removeprefix("torch.")
        cache, loc, nope, rope, _, expected = _host_case(
            padded_loc, NOPE_DIM, ROPE_DIM, torch.bfloat16, loc_dtype
        )
        results.append(
            _run_eager(
                f"set_mla_kv_buffer_triton/bf16/{dtype_name}/eager",
                set_mla_kv_buffer_triton,
                cache,
                loc,
                nope,
                rope,
                expected,
            )
        )

        cache, loc, nope, rope, initial_cache, expected = _host_case(
            padded_loc, NOPE_DIM, ROPE_DIM, torch.bfloat16, loc_dtype
        )
        results.append(
            _run_graph(
                f"set_mla_kv_buffer_triton/bf16/{dtype_name}/hip_graph_replay",
                set_mla_kv_buffer_triton,
                cache,
                loc,
                nope,
                rope,
                initial_cache,
                expected,
            )
        )

        cache, loc, nope, rope, _, expected = _host_case(
            padded_loc, NOPE_DIM, ROPE_DIM, torch.bfloat16, loc_dtype
        )
        with get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0):
            results.append(
                _run_eager(
                    f"set_mla_kv_buffer_dcp_sharded_triton/bf16/{dtype_name}/eager",
                    set_mla_kv_buffer_dcp_sharded_triton,
                    cache,
                    loc,
                    nope,
                    rope,
                    expected,
                )
            )

    cache, loc, nope, rope, _, expected = _host_case(
        padded_loc,
        NOPE_DIM,
        ROPE_DIM,
        torch.uint8,
        torch.int64,
        fp8_dtype=fp8_dtype,
    )
    results.append(
        _run_eager(
            "set_mla_kv_buffer_triton_fp8_quant/e4m3fnuz/eager",
            set_mla_kv_buffer_triton_fp8_quant,
            cache,
            loc,
            nope,
            rope,
            expected,
            fp8_dtype,
        )
    )

    cache, loc, nope, rope, initial_cache, expected = _host_case(
        padded_loc,
        NOPE_DIM,
        ROPE_DIM,
        torch.uint8,
        torch.int64,
        fp8_dtype=fp8_dtype,
    )
    results.append(
        _run_graph(
            "set_mla_kv_buffer_triton_fp8_quant/e4m3fnuz/hip_graph_replay",
            set_mla_kv_buffer_triton_fp8_quant,
            cache,
            loc,
            nope,
            rope,
            initial_cache,
            expected,
            fp8_dtype,
        )
    )

    cache, loc, nope, rope, _, expected = _host_case(
        padded_loc, SCALE_NOPE_DIM, SCALE_ROPE_DIM, torch.float32, torch.int64
    )
    results.append(
        _run_eager(
            "set_mla_kv_scale_buffer_triton/fp32/eager",
            set_mla_kv_scale_buffer_triton,
            cache,
            loc,
            nope,
            rope,
            expected,
        )
    )

    cache, loc, nope, rope, initial_cache, expected = _host_case(
        padded_loc, SCALE_NOPE_DIM, SCALE_ROPE_DIM, torch.float32, torch.int64
    )
    results.append(
        _run_graph(
            "set_mla_kv_scale_buffer_triton/fp32/hip_graph_replay",
            set_mla_kv_scale_buffer_triton,
            cache,
            loc,
            nope,
            rope,
            initial_cache,
            expected,
        )
    )

    report = {
        "determination": "already_fixed_cannot_reproduce",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
            "working_clone": str(REPO_ROOT),
            "python_module_root": str(REPO_ROOT / "python"),
            "pdl_supported": is_arch_support_pdl(),
            "tma_writer_supported": bool(
                can_use_set_mla_kv_buffer(
                    NOPE_DIM * torch.tensor([], dtype=torch.bfloat16).element_size(),
                    ROPE_DIM * torch.tensor([], dtype=torch.bfloat16).element_size(),
                )
            ),
        },
        "padded_loc": padded_loc,
        "reserved_skip_index": 0,
        "results": results,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    failures = [
        result
        for result in results
        if not result["slot_zero_unchanged"]
        or not result["cache_exactly_matches_reference"]
    ]
    if failures:
        raise SystemExit(f"validation failed: {failures}")


if __name__ == "__main__":
    main()
