import json
import subprocess
import sys
import time
from pathlib import Path

import torch


ROWS = 4096
COLS = 4096
SEED = 719885503
SENTINEL = -987654.321


def make_source():
    source = torch.arange(ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS)
    source.mul_(1.0e-5).sub_(1000.0)

    finfo = torch.finfo(torch.float32)
    source[0].fill_(0.0)
    source[1] = -0.0
    source[2].fill_(1.401298464324817e-45)
    source[3].fill_(finfo.max)
    source[4].fill_(-finfo.max)
    source[5].fill_(finfo.tiny)
    source[6].fill_(-finfo.tiny)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    source[7] = torch.randn(COLS, generator=generator, dtype=torch.float32)
    return source


def make_inverse(permutation):
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), dtype=permutation.dtype)
    return inverse


def permutation_cases():
    identity = torch.arange(ROWS, dtype=torch.int64)
    return [
        ("identity", identity),
        ("reverse", torch.flip(identity, dims=(0,))),
        ("rotate_right_one", torch.roll(identity, shifts=1, dims=0)),
        ("rotate_left_one", torch.roll(identity, shifts=-1, dims=0)),
        (
            "seeded_random",
            torch.randperm(
                ROWS, generator=torch.Generator(device="cpu").manual_seed(SEED)
            ),
        ),
    ]


def event_ms(start, end):
    return start.elapsed_time(end)


def main():
    device = torch.device("cuda:0")
    source_cpu = make_source()
    source_gpu = source_cpu.to(device=device, non_blocking=False)

    output_a = torch.full(
        (ROWS, COLS), SENTINEL, dtype=torch.float32, device=device
    )
    output_b = torch.full(
        (ROWS, COLS), SENTINEL, dtype=torch.float32, device=device
    )

    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    inverse_start = torch.cuda.Event(enable_timing=True)
    inverse_end = torch.cuda.Event(enable_timing=True)

    results = []
    torch.cuda.synchronize(device)
    for case_index, (name, permutation_cpu) in enumerate(permutation_cases()):
        inverse_cpu = make_inverse(permutation_cpu)
        inverse_argsort = torch.argsort(permutation_cpu)
        permutation_gpu = permutation_cpu.to(device=device, non_blocking=False)
        inverse_gpu = inverse_cpu.to(device=device, non_blocking=False)

        forward_buffer = output_a if case_index % 2 == 0 else output_b
        inverse_buffer = output_b if case_index % 2 == 0 else output_a
        forward_buffer.fill_(SENTINEL)
        inverse_buffer.fill_(SENTINEL)
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
        case_wall_start = time.perf_counter()

        forward_start.record()
        forward_result = torch.index_select(
            source_gpu, 0, permutation_gpu, out=forward_buffer
        )
        forward_end.record()

        inverse_start.record()
        inverse_result = torch.index_select(
            forward_buffer, 0, inverse_gpu, out=inverse_buffer
        )
        inverse_end.record()
        torch.cuda.synchronize(device)

        case_wall_elapsed = time.perf_counter() - case_wall_start
        allocated_after = torch.cuda.memory_allocated(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)

        forward_reference = source_cpu.index_select(0, permutation_cpu)
        inverse_reference = forward_reference.index_select(0, inverse_cpu)
        forward_cpu = forward_buffer.cpu()
        inverse_cpu_result = inverse_buffer.cpu()

        forward_equal = torch.equal(forward_cpu, forward_reference)
        inverse_equal = torch.equal(inverse_cpu_result, inverse_reference)
        source_restored = torch.equal(inverse_cpu_result, source_cpu)
        forward_max_abs = (
            (forward_cpu - forward_reference).abs().max().item()
        )
        inverse_max_abs = (
            (inverse_cpu_result - inverse_reference).abs().max().item()
        )

        checks = {
            "permutation_valid": bool(
                permutation_cpu.min().item() >= 0
                and permutation_cpu.max().item() < ROWS
                and permutation_cpu.unique().numel() == ROWS
            ),
            "inverse_valid": bool(
                inverse_cpu.min().item() >= 0
                and inverse_cpu.max().item() < ROWS
                and inverse_cpu.unique().numel() == ROWS
            ),
            "inverse_matches_argsort": bool(
                torch.equal(inverse_cpu, inverse_argsort)
            ),
            "forward_exact_reference": forward_equal,
            "inverse_exact_reference": inverse_equal,
            "inverse_restores_source": source_restored,
            "source_finite": bool(torch.isfinite(source_cpu).all()),
            "forward_result_is_output_buffer": bool(
                forward_result.data_ptr() == forward_buffer.data_ptr()
            ),
            "inverse_result_is_output_buffer": bool(
                inverse_result.data_ptr() == inverse_buffer.data_ptr()
            ),
            "no_extra_allocation": allocated_after == allocated_before,
        }
        result = {
            "name": name,
            "all_checks_passed": all(checks.values()),
            "checks": checks,
            "forward_output_buffer": "a" if case_index % 2 == 0 else "b",
            "inverse_output_buffer": "b" if case_index % 2 == 0 else "a",
            "raw": {
                "forward_max_abs_difference": forward_max_abs,
                "inverse_max_abs_difference": inverse_max_abs,
                "permutation_min": int(permutation_cpu.min().item()),
                "permutation_max": int(permutation_cpu.max().item()),
                "inverse_min": int(inverse_cpu.min().item()),
                "inverse_max": int(inverse_cpu.max().item()),
            },
            "timing": {
                "forward_cuda_event_ms": event_ms(forward_start, forward_end),
                "inverse_cuda_event_ms": event_ms(inverse_start, inverse_end),
                "case_wall_seconds": case_wall_elapsed,
                "method": "one CUDA event pair around each index_select call; one wall-clock measurement per case",
            },
            "memory": {
                "allocated_before_bytes": allocated_before,
                "allocated_after_bytes": allocated_after,
                "peak_allocated_bytes": peak_allocated,
            },
        }
        results.append(result)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        torch.index_select(source_gpu, 0, torch.arange(ROWS, device=device), out=output_a)
        torch.cuda.synchronize(device)
    kernel_names = [
        event.name
        for event in profile.events()
        if str(event.device_type).endswith("CUDA")
        and "vectorized_gather_kernel" in event.name
    ]

    properties = torch.cuda.get_device_properties(device)
    record = {
        "label": "persistent-checkout native index_select alternating-output-buffer property",
        "recorded_at_utc": subprocess.check_output(
            ["date", "-u", "+%FT%TZ"], text=True
        ).strip(),
        "operation_contract": {
            "operation": "torch.index_select(source, 0, permutation, out=buffer)",
            "semantics": "buffer[i, :] = source[permutation[i], :]",
            "inverse_semantics": "inverse_buffer[i, :] = forward_buffer[inverse[i], :]",
            "inverse_derivation": "inverse[permutation[i]] = i, cross-checked against torch.argsort(permutation)",
            "dtype": "torch.float32",
            "shape": [ROWS, COLS],
            "output_buffers": ["a", "b"],
            "alternation": "forward and inverse calls alternate between the two preallocated buffers across cases",
        },
        "finite_adversarial_input": {
            "description": "deterministic finite float32 matrix with signed zero, minimum subnormal, minimum normal, positive and negative finite maximum, and seeded random rows",
            "all_finite": bool(torch.isfinite(source_cpu).all()),
        },
        "bounds": {
            "cases": len(results),
            "calls_per_case": 2,
            "profiler_control_calls": 1,
            "full_model_weights": False,
            "unbounded_or_burn_loops": False,
        },
        "environment": {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_module": torch.__file__,
            "hip": torch.version.hip,
            "gpu_name": properties.name,
            "gpu_major_minor": [properties.major, properties.minor],
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_multiprocessor_count": properties.multi_processor_count,
            "required_local_image_id": "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
            "image_id_note": "operator-provided local image ID; no container image API was available for independent in-container verification",
        },
        "vectorized_kernel_observed": bool(kernel_names),
        "vectorized_kernel_names": sorted(set(kernel_names)),
        "results": results,
        "all_cases_passed": all(result["all_checks_passed"] for result in results),
    }

    output_path = Path(__file__).with_name("alternating-index-select-results.json")
    output_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not record["all_cases_passed"] or not record["vectorized_kernel_observed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
