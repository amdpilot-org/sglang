import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import sgl_kernel
import torch


TOPK = 2048
REPEATS = 5


def ordered_key16(score: torch.Tensor) -> torch.Tensor:
    bits = score.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key32 = torch.where(
        (bits & 0x80000000) != 0,
        (~bits) & 0xFFFFFFFF,
        bits | 0x80000000,
    )
    return (key32 >> 16).to(torch.int64)


def coop_candidate_population(score: torch.Tensor) -> dict[str, list[int]]:
    keys = ordered_key16(score)
    kth_index = torch.topk(score, TOPK, dim=1).indices[:, -1:]
    kth_key = keys.gather(1, kth_index)
    round0_bin = kth_key >> 4
    round0_count = (keys >> 4 == round0_bin).sum(dim=1)
    full_key_count = (keys == kth_key).sum(dim=1)
    resolved_count = torch.where(
        round0_count > 4096, full_key_count, round0_count
    )
    return {
        "round0": round0_count.tolist(),
        "resolved": resolved_count.tolist(),
        "overflow_rows": int((resolved_count > 4096).sum().item()),
    }


def fp16_coarse_population(score: torch.Tensor) -> dict[str, list[int]]:
    half_bits = (
        score.to(torch.float16).view(torch.int16).to(torch.int64) & 0xFFFF
    )
    keys = torch.where(
        (half_bits & 0x8000) != 0,
        (~half_bits) & 0xFFFF,
        half_bits | 0x8000,
    )
    bins = keys >> 8
    kth_index = torch.topk(score, TOPK, dim=1).indices[:, -1:]
    kth_bin = bins.gather(1, kth_index)
    count = (bins == kth_bin).sum(dim=1)
    return {
        "round0": count.tolist(),
        "resolved": count.tolist(),
        "overflow_rows": int((count > 6144).sum().item()),
    }


def candidate_population(
    score: torch.Tensor, path: str
) -> dict[str, list[int]]:
    if path == "rocm-coop":
        return coop_candidate_population(score)
    return fp16_coarse_population(score)


def set_mismatch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (
        left.sort(dim=1).values != right.sort(dim=1).values
    ).sum(dim=1)


def structural_metrics(
    indices: torch.Tensor, lengths: torch.Tensor
) -> dict[str, list[int]]:
    invalid = ((indices < 0) | (indices >= lengths[:, None])).sum(dim=1)
    sorted_indices = indices.sort(dim=1).values
    duplicates = (sorted_indices[:, 1:] == sorted_indices[:, :-1]).sum(dim=1)
    return {
        "invalid_indices": invalid.tolist(),
        "duplicate_indices": duplicates.tolist(),
    }


def continuous_metrics(
    score: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, object]:
    oracle = torch.topk(score, TOPK, dim=1)
    oracle_indices = oracle.indices.to(torch.int32)
    selected_values = score.gather(1, indices.long())
    kth_value = oracle.values[:, -1]
    next_value = torch.topk(score, TOPK + 1, dim=1).values[:, -1]
    selected_mask = torch.zeros(
        score.shape, dtype=torch.bool, device=score.device
    )
    selected_mask.scatter_(1, indices.long(), True)

    metrics: dict[str, object] = {
        **structural_metrics(indices, lengths),
        "set_errors_vs_torch": set_mismatch(indices, oracle_indices).tolist(),
        "value_errors_vs_torch": (
            selected_values.sort(dim=1).values
            != oracle.values.sort(dim=1).values
        ).sum(dim=1).tolist(),
        "selected_above_kth": (selected_values > kth_value[:, None])
        .sum(dim=1)
        .tolist(),
        "selected_at_kth": (selected_values == kth_value[:, None])
        .sum(dim=1)
        .tolist(),
        "selected_below_kth": (selected_values < kth_value[:, None])
        .sum(dim=1)
        .tolist(),
        "unselected_above_kth": ((score > kth_value[:, None]) & ~selected_mask)
        .sum(dim=1)
        .tolist(),
        "unselected_at_kth": ((score == kth_value[:, None]) & ~selected_mask)
        .sum(dim=1)
        .tolist(),
        "kth_next_gap": (kth_value - next_value).tolist(),
        "duplicate_source_values": (
            score.sort(dim=1).values.diff(dim=1) == 0
        )
        .sum(dim=1)
        .tolist(),
    }
    return metrics


def ties_metrics(
    score: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, object]:
    oracle = torch.topk(score, TOPK, dim=1)
    oracle_indices = oracle.indices.to(torch.int32)
    selected_values = score.gather(1, indices.long())
    kth_value = oracle.values[:, -1]
    tie_group = score == kth_value[:, None]
    selected_from_tie = tie_group.gather(1, indices.long())
    metrics: dict[str, object] = {
        **structural_metrics(indices, lengths),
        "set_mismatch_vs_torch": set_mismatch(indices, oracle_indices).tolist(),
        "selected_from_tie": selected_from_tie.sum(dim=1).tolist(),
        "selected_at_kth": (selected_values == kth_value[:, None])
        .sum(dim=1)
        .tolist(),
        "selected_below_kth": (selected_values < kth_value[:, None])
        .sum(dim=1)
        .tolist(),
        "tie_group_size": tie_group.sum(dim=1).tolist(),
        "multiple_valid_outputs": bool((tie_group.sum(dim=1) > TOPK).all().item()),
    }
    return metrics


def run_repeats(
    score: torch.Tensor, lengths: torch.Tensor
) -> tuple[list[torch.Tensor], float]:
    outputs: list[torch.Tensor] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        outputs.append(sgl_kernel.fast_topk_v2(score, lengths, TOPK))
    end.record()
    end.synchronize()
    return outputs, start.elapsed_time(end) / REPEATS


def repeat_metrics(
    outputs: list[torch.Tensor], oracle_indices: torch.Tensor
) -> dict[str, object]:
    first = outputs[0]
    return {
        "set_mismatch_vs_repeat0": [
            set_mismatch(output, first).tolist() for output in outputs[1:]
        ],
        "set_mismatch_vs_torch": [
            set_mismatch(output, oracle_indices).tolist() for output in outputs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--path",
        choices=("rocm-coop", "radix-cuda-port"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.path == "rocm-coop":
        tie_capacity = 4096
        score_width = 0.125
        lengths_to_test = (65535, 65536, 65537, 65552, 65553, 73728)
    else:
        tie_capacity = 6144
        score_width = 0.2753143310546875
        lengths_to_test = (65520, 65540, 65550, 65558, 65580, 66000)

    torch.manual_seed(36807)
    torch.cuda.reset_peak_memory_stats()
    continuous_cases: list[dict[str, object]] = []
    for length in lengths_to_test:
        values = torch.linspace(
            1.0,
            1.0 + score_width,
            length,
            dtype=torch.float32,
            device="cuda",
        )
        score = torch.stack(
            [values[torch.randperm(length, device="cuda")] for _ in range(4)]
        )
        lengths = torch.full((4,), length, dtype=torch.int32, device="cuda")
        outputs, milliseconds = run_repeats(score, lengths)
        oracle_indices = torch.topk(score, TOPK, dim=1).indices.to(torch.int32)
        continuous_cases.append(
            {
                "length": length,
                "batch": 4,
                "candidate_population": candidate_population(
                    score, args.path
                ),
                "first_repeat": continuous_metrics(
                    score, outputs[0], lengths
                ),
                **repeat_metrics(outputs, oracle_indices),
                "mean_milliseconds_per_call": milliseconds,
            }
        )
        del score, lengths, outputs, oracle_indices

    score = torch.full(
        (4, 65536), 0.25, dtype=torch.float32, device="cuda"
    )
    lengths = torch.full((4,), 65536, dtype=torch.int32, device="cuda")
    outputs, milliseconds = run_repeats(score, lengths)
    oracle_indices = torch.topk(score, TOPK, dim=1).indices.to(torch.int32)
    ties_control = {
        "length": 65536,
        "batch": 4,
        "candidate_population": candidate_population(score, args.path),
        "first_repeat": ties_metrics(score, outputs[0], lengths),
        **repeat_metrics(outputs, oracle_indices),
        "mean_milliseconds_per_call": milliseconds,
    }
    del score, lengths, outputs, oracle_indices

    continuous_exact = all(
        max(case["first_repeat"]["set_errors_vs_torch"]) == 0
        and max(case["first_repeat"]["value_errors_vs_torch"]) == 0
        and max(case["first_repeat"]["invalid_indices"]) == 0
        and max(case["first_repeat"]["duplicate_indices"]) == 0
        for case in continuous_cases
    )
    ties_valid = (
        max(ties_control["first_repeat"]["invalid_indices"]) == 0
        and max(ties_control["first_repeat"]["duplicate_indices"]) == 0
        and max(ties_control["first_repeat"]["selected_from_tie"]) == TOPK
        and max(ties_control["first_repeat"]["selected_below_kth"]) == 0
        and ties_control["first_repeat"]["multiple_valid_outputs"]
    )
    result = {
        "label": args.label,
        "revision": args.revision,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "sgl_kernel_python": sgl_kernel.__file__,
            "sgl_kernel_native": sgl_kernel.common_ops.__file__,
            "row_split_env": __import__("os").environ.get(
                "SGL_DSA_TOPK_ROW_SPLIT"
            ),
        },
        "continuous_cases": continuous_cases,
        "ties_only_control": ties_control,
        "summary": {
            "continuous_exact": continuous_exact,
            "ties_only_valid": ties_valid,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "repeats": REPEATS,
            "tie_capacity": tie_capacity,
            "path": args.path,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
