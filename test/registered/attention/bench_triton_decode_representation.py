"""Bounded representation benchmark for Triton grouped decode attention."""

import argparse
import json

import torch
from triton import knobs

from test_triton_decode_representation import (
    _make_case,
    _reference,
    _run,
    _sentinel_output,
)


SEQUENCE_LENGTHS = (128, 1024)
REPRESENTATIONS = ("unpacked", "packed_kv", "strided_q")
WARMUP_ITERATIONS = 3
TIMED_ITERATIONS = 10


def _strided_q(case):
    storage = torch.randn(
        case["q"].shape[0],
        2 * case["q"].shape[1],
        case["q"].shape[2],
        dtype=case["q"].dtype,
        device=case["q"].device,
    )
    updated = dict(case)
    updated["q"] = storage[:, ::2, :]
    return updated


def _record_launches(function):
    launches = []

    def enter(metadata):
        launches.append(metadata.get())

    knobs.runtime.launch_enter_hook.add(enter)
    try:
        function()
        torch.cuda.synchronize()
    finally:
        knobs.runtime.launch_enter_hook.remove(enter)
    return launches


def _benchmark(representation, sequence_length):
    case = _make_case(
        representation == "packed_kv",
        seed=2271 + sequence_length,
        sequence_length=sequence_length,
    )
    if representation == "strided_q":
        case = _strided_q(case)

    storage, output = _sentinel_output()
    launches = _record_launches(lambda: _run(case, output))
    reference = _reference(
        case["q"],
        case["k_buffer"],
        case["v_buffer"],
        case["kv_indptr"],
        case["kv_indices"],
    )
    max_abs_error = (output.float() - reference.float()).abs().max().item()
    if not torch.allclose(output.float(), reference.float(), atol=1e-2, rtol=1e-2):
        raise AssertionError(
            f"{representation}/{sequence_length}: max_abs_error={max_abs_error}"
        )
    if not (
        torch.all(storage[:32] == -12345.0)
        and torch.all(storage[-32:] == -12345.0)
    ):
        raise AssertionError(f"{representation}/{sequence_length}: guard overwritten")

    for _ in range(WARMUP_ITERATIONS):
        _run(case, output)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERATIONS):
        _run(case, output)
    end.record()
    torch.cuda.synchronize()

    return {
        "representation": representation,
        "sequence_length": sequence_length,
        "max_abs_error": max_abs_error,
        "mean_ms": start.elapsed_time(end) / TIMED_ITERATIONS,
        "native_dispatches": [
            {"name": launch["name"], "stream": launch["stream"]}
            for launch in launches
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = []
    for sequence_length in SEQUENCE_LENGTHS:
        for representation in REPRESENTATIONS:
            results.append(_benchmark(representation, sequence_length))

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "timing_method": "CUDA events around each bounded batch of calls",
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
