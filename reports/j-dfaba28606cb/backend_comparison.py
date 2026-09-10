#!/usr/bin/env python3
"""Bounded MI300X speculative verifier backend comparison."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    branching: int
    depth: int


CASES = (
    Case("small_tree", 1, 2, 2),
    Case("batch_tree", 8, 2, 3),
    Case("wide_tree", 16, 3, 2),
    Case("deep_tree", 32, 2, 4),
    Case("large_tree", 64, 2, 3),
    Case("max_batch_tree", 128, 2, 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measured-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--image-id",
        default="sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
    )
    return parser.parse_args()


def tree_metadata(
    batch_size: int, branching: int, depth: int, device: torch.device
) -> tuple[int, dict[str, torch.Tensor]]:
    level_sizes = [branching**level for level in range(depth + 1)]
    level_starts = []
    start = 0
    for size in level_sizes:
        level_starts.append(start)
        start += size
    node_count = start

    retrieve_index = (
        torch.arange(batch_size * node_count, dtype=torch.int64, device=device)
        .view(batch_size, node_count)
    )
    retrieve_next_token = torch.full(
        (batch_size, node_count), -1, dtype=torch.int64, device=device
    )
    retrieve_next_sibling = torch.full(
        (batch_size, node_count), -1, dtype=torch.int64, device=device
    )
    parent = torch.full((batch_size, node_count), -1, dtype=torch.int64, device=device)

    for level in range(depth):
        level_size = level_sizes[level]
        child_start = level_starts[level + 1]
        for node_offset in range(level_size):
            node = level_starts[level] + node_offset
            first_child = child_start + node_offset * branching
            retrieve_next_token[:, node] = first_child
            for child_offset in range(branching - 1):
                child = first_child + child_offset
                retrieve_next_sibling[:, child] = child + 1
            for child_offset in range(branching):
                parent[:, first_child + child_offset] = node

    return node_count, {
        "retrieve_index": retrieve_index.contiguous(),
        "retrieve_next_token": retrieve_next_token.contiguous(),
        "retrieve_next_sibling": retrieve_next_sibling.contiguous(),
        "parent": parent.contiguous(),
    }


def reference_verify(
    candidates: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    target_predict: torch.Tensor,
    num_spec_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidates_cpu = candidates.cpu()
    retrieve_index = metadata["retrieve_index"].cpu()
    retrieve_next_token = metadata["retrieve_next_token"].cpu()
    retrieve_next_sibling = metadata["retrieve_next_sibling"].cpu()
    target_flat = target_predict.cpu().reshape(-1)
    batch_size, node_count = candidates_cpu.shape

    predicts = torch.full((batch_size * node_count,), -1, dtype=torch.int32)
    accept_index = torch.full((batch_size, num_spec_steps), -1, dtype=torch.int32)
    accept_token_num = torch.zeros(batch_size, dtype=torch.int32)

    for batch in range(batch_size):
        last_accept_index = int(retrieve_index[batch, 0])
        accept_index[batch, 0] = last_accept_index
        num_correct = 0
        current = 0
        for _ in range(1, num_spec_steps):
            current = int(retrieve_next_token[batch, current])
            while current != -1:
                draft_index = int(retrieve_index[batch, current])
                draft_token = int(candidates_cpu[batch, current])
                target_token = int(target_flat[last_accept_index])
                if draft_token == target_token:
                    predicts[last_accept_index] = target_token
                    num_correct += 1
                    accept_index[batch, num_correct] = draft_index
                    last_accept_index = draft_index
                    break
                current = int(retrieve_next_sibling[batch, current])
            if current == -1:
                break
        accept_token_num[batch] = num_correct
        predicts[last_accept_index] = int(target_flat[last_accept_index])

    return predicts, accept_index, accept_token_num


def allocate_outputs(
    batch_size: int, node_count: int, num_spec_steps: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "predicts": torch.full(
            (batch_size * node_count,), -1, dtype=torch.int32, device=device
        ),
        "accept_index": torch.full(
            (batch_size, num_spec_steps), -1, dtype=torch.int32, device=device
        ),
        "accept_token_num": torch.zeros(batch_size, dtype=torch.int32, device=device),
    }


def run_native(
    case_data: dict[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    from sgl_kernel import verify_tree_greedy

    outputs = allocate_outputs(
        int(case_data["batch_size"]),
        int(case_data["node_count"]),
        int(case_data["num_spec_steps"]),
        device,
    )
    verify_tree_greedy(
        predicts=outputs["predicts"],
        accept_index=outputs["accept_index"],
        accept_token_num=outputs["accept_token_num"],
        candidates=case_data["candidates"],
        retrive_index=case_data["metadata"]["retrieve_index"],
        retrive_next_token=case_data["metadata"]["retrieve_next_token"],
        retrive_next_sibling=case_data["metadata"]["retrieve_next_sibling"],
        target_predict=case_data["target_predict"],
    )
    return outputs


def run_triton(
    case_data: dict[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_triton

    outputs = allocate_outputs(
        int(case_data["batch_size"]),
        int(case_data["node_count"]),
        int(case_data["num_spec_steps"]),
        device,
    )
    verify_tree_greedy_triton(
        predicts=outputs["predicts"],
        accept_index=outputs["accept_index"],
        accept_token_num=outputs["accept_token_num"],
        candidates=case_data["candidates"],
        retrieve_index=case_data["metadata"]["retrieve_index"],
        retrieve_next_token=case_data["metadata"]["retrieve_next_token"],
        retrieve_next_sibling=case_data["metadata"]["retrieve_next_sibling"],
        target_predict=case_data["target_predict"],
    )
    return outputs


def timing_summary(milliseconds: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "stddev_ms": statistics.pstdev(milliseconds),
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
        "runs": len(milliseconds),
    }


def time_backend(
    runner: Callable[[dict[str, object], torch.device], dict[str, torch.Tensor]],
    case_data: dict[str, object],
    device: torch.device,
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, object]:
    for _ in range(warmup_runs):
        runner(case_data, device)
    torch.cuda.synchronize()

    timings = []
    for _ in range(measured_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner(case_data, device)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return timing_summary(timings)


def output_difference(
    output: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, object]:
    differences = {}
    for key in ("predicts", "accept_index", "accept_token_num"):
        difference = (output[key].cpu() - reference[key]).abs()
        differences[key] = {
            "exact_match": bool(torch.equal(output[key].cpu(), reference[key])),
            "max_abs_difference": int(difference.max().item()),
            "mismatched_elements": int((difference != 0).sum().item()),
        }
    return differences


def probe_stochastic_native_op(device: torch.device) -> dict[str, object]:
    node_count, metadata = tree_metadata(1, 2, 2, device)
    candidates = torch.randint(0, 32, (1, node_count), dtype=torch.int64, device=device)
    target_probs = torch.rand((1, node_count, 32), dtype=torch.float32, device=device)
    draft_probs = torch.rand_like(target_probs)
    outputs = allocate_outputs(1, node_count, 3, device)
    try:
        from sgl_kernel import tree_speculative_sampling_target_only

        tree_speculative_sampling_target_only(
            predicts=outputs["predicts"],
            accept_index=outputs["accept_index"],
            accept_token_num=outputs["accept_token_num"],
            candidates=candidates,
            retrive_index=metadata["retrieve_index"],
            retrive_next_token=metadata["retrieve_next_token"],
            retrive_next_sibling=metadata["retrieve_next_sibling"],
            uniform_samples=torch.rand((1, node_count), device=device),
            uniform_samples_for_final_sampling=torch.rand(1, device=device),
            target_probs=target_probs,
            draft_probs=draft_probs,
        )
    except Exception as error:
        return {
            "available": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {"available": True}


def native_module_paths() -> dict[str, str]:
    import sgl_kernel

    package_root = Path(sgl_kernel.__file__).parent
    native_files = sorted(package_root.glob("*.so"))
    return {
        "sgl_kernel_package": str(sgl_kernel.__file__),
        "sgl_kernel_speculative": str(package_root / "speculative.py"),
        "native_extension": str(native_files[0]) if native_files else "not_found",
    }


def triton_paths() -> dict[str, str]:
    import triton
    from sglang.kernels.ops.speculative.spec_tree import (
        verify_tree_greedy_kernel_triton,
    )

    return {
        "triton_package": str(Path(triton.__file__).parent),
        "triton_version": triton.__version__,
        "kernel_module": verify_tree_greedy_kernel_triton.__module__,
        "kernel_source": inspect.getsourcefile(verify_tree_greedy_kernel_triton.fn)
        or "unknown",
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", "default"),
    }


def gpu_identity() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    try:
        gfx = subprocess.check_output(
            ["rocm-smi", "--showid"], text=True, timeout=10
        )
    except Exception:
        gfx = "unavailable"
    return {
        "name": properties.name,
        "total_bytes": properties.total_memory,
        "torch_hip_version": torch.version.hip,
        "rocm_smi_id": gfx.strip(),
    }


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0 or args.measured_runs <= 0:
        raise ValueError("warmup runs must be nonnegative and measured runs positive")

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    hidden_size = 256
    vocab_size = 4096
    weight = torch.randn(hidden_size, vocab_size, device=device, dtype=torch.float32)
    weight_bytes = weight.numel() * weight.element_size()
    if weight_bytes >= 4 * 1024**3:
        raise ValueError("synthetic weight exceeds the 4 GiB limit")

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parents[2]
    ).strip()
    results: dict[str, object] = {
        "label": "checkout backend comparison; installed-source baseline is separate",
        "source_commit": source_commit,
        "image_identity": args.image_id,
        "container_hostname": os.uname().nodename,
        "gpu": gpu_identity(),
        "torch": {
            "version": torch.__version__,
            "file": torch.__file__,
            "hip": torch.version.hip,
        },
        "native_paths": native_module_paths(),
        "triton_paths": triton_paths(),
        "backends": {
            "native_hip": "sgl_kernel.verify_tree_greedy",
            "triton": "sglang...spec_tree.verify_tree_greedy_kernel_triton",
        },
        "synthetic_weights": {
            "shape": [hidden_size, vocab_size],
            "dtype": "float32",
            "bytes": weight_bytes,
            "under_4_gib": True,
            "generation": "torch.randn on device with fixed seed",
        },
        "reference": "independent Python CPU traversal of the same draft tree",
        "accuracy_gate": "exact equality for predicts, accept_index, and accept_token_num",
        "timing_method": {
            "event": "torch.cuda.Event(enable_timing=True)",
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "scope": "fresh output allocation, backend dispatch, kernel execution, and synchronize",
            "bounded": True,
        },
        "cases": [],
        "unsupported_native_stochastic_sampling": probe_stochastic_native_op(device),
        "command": " ".join(sys.argv),
    }

    for case in CASES:
        torch.cuda.reset_peak_memory_stats()
        node_count, metadata = tree_metadata(
            case.batch_size, case.branching, case.depth, device
        )
        num_spec_steps = case.depth + 1
        features = torch.randn(
            (case.batch_size * node_count, hidden_size),
            device=device,
            dtype=torch.float32,
        )
        logits = features @ weight
        target_probs = torch.softmax(logits, dim=-1)
        target_predict = torch.argmax(target_probs, dim=-1).view(
            case.batch_size, node_count
        )
        candidates = torch.randint(
            0, vocab_size, (case.batch_size, node_count), dtype=torch.int64, device=device
        )
        parent_target = torch.where(
            metadata["parent"] >= 0,
            target_predict.gather(1, metadata["parent"].clamp_min(0)),
            target_predict,
        )
        match_mask = (
            torch.arange(case.batch_size * node_count, device=device).view(
                case.batch_size, node_count
            )
            % 2
            == 0
        )
        candidates = torch.where(match_mask, parent_target, candidates)
        case_data = {
            "batch_size": case.batch_size,
            "node_count": node_count,
            "num_spec_steps": num_spec_steps,
            "branching": case.branching,
            "depth": case.depth,
            "candidates": candidates,
            "metadata": metadata,
            "target_predict": target_predict,
        }

        reference = dict(
            zip(
                ("predicts", "accept_index", "accept_token_num"),
                reference_verify(candidates, metadata, target_predict, num_spec_steps),
            )
        )
        native_output = run_native(case_data, device)
        triton_output = run_triton(case_data, device)
        native_difference = output_difference(native_output, reference)
        triton_difference = output_difference(triton_output, reference)
        backend_difference = output_difference(native_output, {
            "predicts": triton_output["predicts"].cpu(),
            "accept_index": triton_output["accept_index"].cpu(),
            "accept_token_num": triton_output["accept_token_num"].cpu(),
        })

        native_timing = time_backend(
            run_native, case_data, device, args.warmup_runs, args.measured_runs
        )
        triton_timing = time_backend(
            run_triton, case_data, device, args.warmup_runs, args.measured_runs
        )
        case_result = {
            "name": case.name,
            "batch_size": case.batch_size,
            "branching": case.branching,
            "depth": case.depth,
            "node_count": node_count,
            "num_spec_steps": num_spec_steps,
            "dtypes": {
                "candidates": "int64",
                "tree_metadata": "int64",
                "target_predict": "int64",
                "outputs": "int32",
                "target_probs": "float32",
            },
            "target_probability_bytes": target_probs.numel()
            * target_probs.element_size(),
            "target_probability_sum_error": float(
                (target_probs.sum(dim=-1) - 1).abs().max().item()
            ),
            "reference_accept_token_num": reference["accept_token_num"].tolist(),
            "native_vs_reference": native_difference,
            "triton_vs_reference": triton_difference,
            "native_vs_triton": backend_difference,
            "native_timing_ms": native_timing,
            "triton_timing_ms": triton_timing,
            "native_to_triton_median_speedup": triton_timing["median_ms"]
            / native_timing["median_ms"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        results["cases"].append(case_result)

        for backend_difference_values in (
            native_difference,
            triton_difference,
            backend_difference,
        ):
            if not all(value["exact_match"] for value in backend_difference_values.values()):
                raise RuntimeError(f"accuracy gate failed for {case.name}")

    results["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
