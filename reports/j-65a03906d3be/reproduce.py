import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path


START = time.perf_counter()

import torch
import triton

from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter


DEVICE = torch.device("cuda")
SENTINEL_INDEX = -2147483648
SENTINEL_BF16 = -12345.0
SENTINEL_FP8_BYTE = 0x7F
SENTINEL_SCALE = float("nan")


def bit_source(rows, columns, dtype, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.rand((rows, columns), generator=generator, dtype=torch.float32) * 2 - 1
    if dtype == torch.bfloat16:
        special = torch.tensor(
            [
                0x0000,
                0x8000,
                0x0080,
                0x8080,
                0x3F80,
                0xBF80,
                0x7F7F,
                0xFF7F,
            ],
            dtype=torch.uint16,
        ).view(torch.int16)
        bits = random.to(torch.bfloat16).view(torch.int16).reshape(-1)
    elif dtype == torch.float8_e4m3fn:
        special = torch.tensor(
            [0x00, 0x80, 0x01, 0x81, 0x38, 0xB8, 0x7E, 0xFE],
            dtype=torch.uint8,
        )
        bits = random.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
    else:
        raise ValueError(f"unsupported source dtype: {dtype}")
    positions = torch.arange(rows * columns) % special.numel()
    bits = special[positions].reshape(rows, columns).contiguous()
    return bits.to(DEVICE).view(dtype)


def scale_source(rows, width, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.rand((rows, width), generator=generator, dtype=torch.float32) * 2 - 1
    special = torch.tensor(
        [0.0, -0.0, 0.125, 1.0, -1.0, 2.0, -2.0, 0.5],
        dtype=torch.float32,
    )
    positions = torch.arange(rows * width) % special.numel()
    values = special[positions].reshape(rows, width)
    values = torch.where(random < 0, values, random)
    return values.to(DEVICE)


def guarded_1d(length, dtype, sentinel):
    guard = 32
    storage = torch.full((guard + length + guard,), sentinel, device=DEVICE, dtype=dtype)
    return storage, guard, length, guard


def guarded_2d(rows, columns, dtype, sentinel):
    guard = 8
    storage = torch.full(
        (guard + rows + guard, columns), sentinel, device=DEVICE, dtype=dtype
    )
    return storage, guard, rows, guard


def make_case(case):
    num_experts = len(case["padded_counts"])
    total_padded = sum(case["padded_counts"])
    total_valid = sum(case["valid_counts"])
    topk = case["topk"]
    assert total_valid % topk == 0
    tokens = total_valid // topk
    hidden = case["hidden_size"]

    local_ids = torch.cat(
        [
            torch.full((count,), expert, dtype=torch.int32)
            for expert, count in enumerate(case["valid_counts"])
        ]
    )
    global_ids = local_ids + case["expert_start"]
    topk_ids = global_ids.reshape(tokens, topk).to(DEVICE).contiguous()
    source = bit_source(tokens, hidden, case["dtype"], case["seed"])

    use_scale = case["use_scale"]
    scale_width = hidden // 128
    scale = scale_source(tokens, scale_width, case["seed"] + 1) if use_scale else None

    counts = torch.tensor(case["padded_counts"], dtype=torch.int32, device=DEVICE)
    valid = torch.tensor(case["valid_counts"], dtype=torch.int32, device=DEVICE)
    starts = torch.full((num_experts,), SENTINEL_INDEX, dtype=torch.int32, device=DEVICE)

    m_storage, m_guard, m_rows, _ = guarded_1d(total_padded, torch.int32, SENTINEL_INDEX)
    m_indices = m_storage[m_guard : m_guard + m_rows]
    index_storage, index_guard, index_rows, _ = guarded_1d(
        tokens * topk, torch.int32, SENTINEL_INDEX
    )
    output_index = index_storage[index_guard : index_guard + index_rows].view(tokens, topk)

    if case["dtype"] == torch.bfloat16:
        tensor_storage, tensor_guard, tensor_rows, _ = guarded_2d(
            total_padded, hidden, torch.bfloat16, SENTINEL_BF16
        )
        output_tensor = tensor_storage[tensor_guard : tensor_guard + tensor_rows]
    else:
        tensor_storage = torch.empty(
            (8 + total_padded + 8, hidden), device=DEVICE, dtype=torch.uint8
        )
        tensor_storage.fill_(SENTINEL_FP8_BYTE)
        output_tensor = tensor_storage[8 : 8 + total_padded].view(torch.float8_e4m3fn)
        tensor_guard = 8
        tensor_rows = total_padded

    if use_scale:
        scale_storage, scale_guard, scale_rows, _ = guarded_2d(
            total_padded, scale_width, torch.float32, SENTINEL_SCALE
        )
        output_scale = scale_storage[scale_guard : scale_guard + scale_rows]
    else:
        scale_storage = None
        output_scale = None

    return {
        **case,
        "tokens": tokens,
        "total_padded": total_padded,
        "source": source,
        "scale": scale,
        "topk_ids": topk_ids,
        "local_ids": local_ids,
        "counts": counts,
        "valid": valid,
        "starts": starts,
        "m_storage": m_storage,
        "m_indices": m_indices,
        "index_storage": index_storage,
        "output_index": output_index,
        "tensor_storage": tensor_storage,
        "output_tensor": output_tensor,
        "scale_storage": scale_storage,
        "output_scale": output_scale,
    }


def reference(case):
    padded = torch.tensor(case["padded_counts"], dtype=torch.int32)
    valid = torch.tensor(case["valid_counts"], dtype=torch.int32)
    starts = torch.cumsum(padded, 0) - padded
    expected_m = torch.cat(
        [
            torch.cat(
                [
                    torch.full((int(valid_count),), expert, dtype=torch.int32),
                    torch.full(
                        (int(padded_count - valid_count),), -1, dtype=torch.int32
                    ),
                ]
            )
            for expert, (valid_count, padded_count) in enumerate(zip(valid, padded))
        ]
    )

    tokens = case["tokens"]
    topk = case["topk"]
    expected_destinations = torch.cat(
        [
            starts[expert] + torch.arange(int(valid_count), dtype=torch.int32)
            for expert, valid_count in enumerate(valid)
        ]
    ).sort()[0]
    total_padded = case["total_padded"]
    hidden = case["hidden_size"]
    return starts, expected_m, expected_destinations


def validate(case):
    starts, expected_m, expected_destinations = reference(case)
    expected_post_scatter_starts = starts + torch.tensor(
        case["valid_counts"], dtype=torch.int32
    )
    actual_destinations = case["output_index"].reshape(-1).cpu()
    local_ids = case["topk_ids"].cpu() - case["expert_start"]
    source_bits = case["source"].cpu()
    source_bits = (
        source_bits.view(torch.int16)
        if case["dtype"] == torch.bfloat16
        else source_bits.view(torch.uint8)
    )
    source_per_assignment = (
        source_bits.unsqueeze(1)
        .expand(case["tokens"], case["topk"], case["hidden_size"])
        .reshape(-1, case["hidden_size"])
    )
    scattered_bits = case["output_tensor"][actual_destinations.to(DEVICE)].cpu()
    scattered_bits = (
        scattered_bits.view(torch.int16)
        if case["dtype"] == torch.bfloat16
        else scattered_bits.view(torch.uint8)
    )
    checks = {
        "post_scatter_starts": torch.equal(
            case["starts"].cpu(), expected_post_scatter_starts
        ),
        "m_indices": torch.equal(case["m_indices"].cpu(), expected_m),
        "output_index_assigned": bool((actual_destinations >= 0).all()),
        "output_index_unique": actual_destinations.unique().numel()
        == actual_destinations.numel(),
        "output_index_destination_set": torch.equal(
            actual_destinations.sort()[0], expected_destinations
        ),
        "output_index_maps_to_expert": torch.equal(
            case["m_indices"][actual_destinations.to(DEVICE)].cpu(), local_ids.reshape(-1)
        ),
    }
    if case["dtype"] == torch.bfloat16:
        checks["output_tensor_bits_per_assignment"] = torch.equal(
            scattered_bits, source_per_assignment
        )
    else:
        checks["output_tensor_bits_per_assignment"] = torch.equal(
            scattered_bits, source_per_assignment
        )
    if case["use_scale"] and case["dtype"] != torch.bfloat16:
        scale_bits = case["scale"].cpu().view(torch.int32)
        scale_per_assignment = (
            scale_bits.unsqueeze(1)
            .expand(case["tokens"], case["topk"], case["scale"].shape[1])
            .reshape(-1, case["scale"].shape[1])
        )
        scattered_scale_bits = (
            case["output_scale"][actual_destinations.to(DEVICE)].cpu().view(torch.int32)
        )
        checks["output_scale_bits_per_assignment"] = torch.equal(
            scattered_scale_bits, scale_per_assignment
        )
    checks["m_guard_before"] = bool(
        torch.all(case["m_storage"][:32].cpu() == SENTINEL_INDEX)
    )
    checks["m_guard_after"] = bool(
        torch.all(case["m_storage"][32 + case["total_padded"] :].cpu() == SENTINEL_INDEX)
    )
    checks["index_guard_before"] = bool(
        torch.all(case["index_storage"][:32].cpu() == SENTINEL_INDEX)
    )
    checks["index_guard_after"] = bool(
        torch.all(
            case["index_storage"][32 + case["tokens"] * case["topk"] :].cpu()
            == SENTINEL_INDEX
        )
    )
    if case["dtype"] == torch.bfloat16:
        sentinel_bits = torch.tensor(SENTINEL_BF16, dtype=torch.bfloat16).view(torch.int16)
        checks["tensor_guard_before"] = bool(
            torch.all(case["tensor_storage"][:8].cpu().view(torch.int16) == sentinel_bits)
        )
        checks["tensor_guard_after"] = bool(
            torch.all(
                case["tensor_storage"][8 + case["total_padded"] :].cpu().view(torch.int16)
                == sentinel_bits
            )
        )
    else:
        checks["tensor_guard_before"] = bool(
            torch.all(case["tensor_storage"][:8].cpu() == SENTINEL_FP8_BYTE)
        )
        checks["tensor_guard_after"] = bool(
            torch.all(
                case["tensor_storage"][8 + case["total_padded"] :].cpu()
                == SENTINEL_FP8_BYTE
            )
        )
    if case["use_scale"]:
        checks["scale_guard_before"] = bool(
            torch.all(
                case["scale_storage"][:8].cpu().view(torch.int32)
                == torch.tensor(SENTINEL_SCALE).view(torch.int32)
            )
        )
        checks["scale_guard_after"] = bool(
            torch.all(
                case["scale_storage"][8 + case["total_padded"] :].cpu().view(torch.int32)
                == torch.tensor(SENTINEL_SCALE).view(torch.int32)
            )
        )
    return checks


def launch(case):
    ep_scatter(
        case["source"],
        case["scale"],
        case["topk_ids"],
        case["counts"],
        case["valid"],
        case["starts"],
        case["output_tensor"],
        case["output_scale"],
        case["m_indices"],
        case["output_index"],
        expert_start=case["expert_start"],
    )


def time_case(case):
    for _ in range(3):
        launch(case)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(10):
        start_event.record()
        launch(case)
        end_event.record()
        torch.cuda.synchronize()
        samples.append(start_event.elapsed_time(end_event))
    return samples


def graph_control(case):
    addresses = {
        name: tensor.data_ptr()
        for name, tensor in (
            ("source", case["source"]),
            ("output_tensor", case["output_tensor"]),
            ("m_indices", case["m_indices"]),
            ("output_index", case["output_index"]),
        )
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch(case)
        launch(case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            launch(case)
        graph.replay()
        torch.cuda.synchronize()
        replay_addresses = {
            name: tensor.data_ptr()
            for name, tensor in (
                ("source", case["source"]),
                ("output_tensor", case["output_tensor"]),
                ("m_indices", case["m_indices"]),
                ("output_index", case["output_index"]),
            )
        }
        checks = validate(case)
        return {
            "supported": True,
            "static_addresses_preserved": addresses == replay_addresses,
            "addresses_before": addresses,
            "addresses_after": replay_addresses,
            "checks": checks,
            "passed": addresses == replay_addresses and all(checks.values()),
        }
    except Exception as error:
        return {
            "supported": False,
            "error": f"{type(error).__name__}: {error}",
            "addresses_before": addresses,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/ep-scatter-dtype-gfx942-results.json")
    parser.add_argument("--start-ns", type=int, default=None)
    args = parser.parse_args()

    timed_cases = [
        {
            "name": "bf16_hidden128",
            "dtype": torch.bfloat16,
            "use_scale": False,
            "hidden_size": 128,
            "topk": 2,
            "padded_counts": [128, 256, 0, 128],
            "valid_counts": [100, 200, 0, 128],
            "expert_start": 0,
            "seed": 0x65A039,
        },
        {
            "name": "bf16_hidden1024",
            "dtype": torch.bfloat16,
            "use_scale": False,
            "hidden_size": 1024,
            "topk": 2,
            "padded_counts": [128, 256, 0, 128],
            "valid_counts": [100, 200, 0, 128],
            "expert_start": 0,
            "seed": 0x65A03A,
        },
        {
            "name": "fp8_hidden128",
            "dtype": torch.float8_e4m3fn,
            "use_scale": True,
            "hidden_size": 128,
            "topk": 2,
            "padded_counts": [128, 256, 0, 128],
            "valid_counts": [100, 200, 0, 128],
            "expert_start": 0,
            "seed": 0x65A03B,
        },
        {
            "name": "fp8_hidden1024",
            "dtype": torch.float8_e4m3fn,
            "use_scale": True,
            "hidden_size": 1024,
            "topk": 2,
            "padded_counts": [128, 256, 0, 128],
            "valid_counts": [100, 200, 0, 128],
            "expert_start": 0,
            "seed": 0x65A03C,
        },
    ]

    results = []
    first_elapsed = None
    for definition in timed_cases:
        case = make_case(definition)
        launch(case)
        torch.cuda.synchronize()
        if first_elapsed is None:
            first_elapsed = time.perf_counter() - START
        checks = validate(case)
        samples = time_case(case)
        results.append(
            {
                "name": definition["name"],
                "dtype": str(definition["dtype"]).replace("torch.", ""),
                "use_scale": definition["use_scale"],
                "hidden_size": definition["hidden_size"],
                "tokens": case["tokens"],
                "topk": definition["topk"],
                "padded_counts": definition["padded_counts"],
                "valid_counts": definition["valid_counts"],
                "expert_start": definition["expert_start"],
                "checks": checks,
                "passed": all(checks.values()),
                "timing_samples_ms": samples,
                "timing_median_ms": sorted(samples)[len(samples) // 2],
            }
        )

    ignored_scale_definition = {
        **timed_cases[0],
        "name": "bf16_scale_is_ignored",
        "use_scale": True,
    }
    ignored_case = make_case(ignored_scale_definition)
    launch(ignored_case)
    torch.cuda.synchronize()
    ignored_checks = validate(ignored_case)
    ignored_checks["output_scale_remains_sentinel"] = bool(
        torch.all(
            ignored_case["output_scale"].cpu().view(torch.int32)
            == torch.tensor(SENTINEL_SCALE).view(torch.int32)
        )
    )

    mismatch_definition = {
        **timed_cases[2],
        "name": "fp8_scale_dtype_mismatch",
    }
    mismatch_case = make_case(mismatch_definition)
    mismatch_case["output_scale"] = mismatch_case["output_scale"].to(torch.float16)
    mismatch_error = None
    try:
        launch(mismatch_case)
    except AssertionError as error:
        mismatch_error = str(error)

    graph_result = graph_control(make_case(timed_cases[0]))

    properties = torch.cuda.get_device_properties(0)
    torch_root = Path(torch.__file__).resolve().parent
    native_hsacos = sorted(glob.glob(os.environ.get("TRITON_CACHE_DIR", "") + "/**/*.hsaco", recursive=True))
    record = {
        "label": "mirror PR 241 fixed-commit direct EP scatter dtype matrix",
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "process_start_perf_seconds": START,
        "start_ns": args.start_ns,
        "first_gpu_execution_elapsed_seconds": first_elapsed,
        "commands": [
            "TRITON_CACHE_DIR=/tmp/sglang-cache-j-65a03906d3be/triton-fixed PYTHONPATH=/job/sglang/python /opt/venv/bin/python reports/j-65a03906d3be/reproduce.py --output /tmp/ep-scatter-dtype-gfx942-results.json"
        ],
        "python": "/opt/venv/bin/python",
        "torch": {
            "version": torch.__version__,
            "file": torch.__file__,
            "native": str(torch_root / "_C.cpython-310-x86_64-linux-gnu.so"),
            "hip": str(torch_root / "lib/libtorch_hip.so"),
            "hip_version": str(torch.version.hip),
        },
        "triton": {"version": triton.__version__, "file": triton.__file__},
        "sglang": {
            "module": "/job/sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py",
            "commit": subprocess.check_output(
                ["git", "-C", "/job/sglang", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_major_minor": [properties.major, properties.minor],
            "multi_processor_count": properties.multi_processor_count,
        },
        "image": {
            "required": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "operator_provided_local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        },
        "native_dispatch": {
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "hsaco_paths": native_hsacos,
        },
        "reference": "independent CPU prefix/scatter construction over exact BF16/FP8 payload bits and FP32 scale bits; sentinel guards around all outputs",
        "timing_method": "3 warmups then 10 CUDA-event samples per timed case; median reported",
        "timed_cases": results,
        "all_timed_cases_passed": all(result["passed"] for result in results),
        "bf16_scale_is_ignored": {
            "checks": ignored_checks,
            "passed": all(ignored_checks.values()),
        },
        "fp8_scale_dtype_mismatch": {
            "failed_clearly": mismatch_error is not None,
            "error": mismatch_error,
        },
        "static_graph_address_control": graph_result,
    }
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not record["all_timed_cases_passed"]:
        raise SystemExit(1)
    if not record["bf16_scale_is_ignored"]["passed"]:
        raise SystemExit(1)
    if not record["fp8_scale_dtype_mismatch"]["failed_clearly"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
