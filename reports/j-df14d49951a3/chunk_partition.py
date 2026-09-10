#!/usr/bin/env python3
"""Bounded gfx942 chunk-partition study for installed speculative GPU kernels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from sgl_kernel import build_tree_kernel_efficient, verify_tree_greedy


IMAGE_ID = (
    "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909 "
    "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1"
)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def build_tree(
    batch_size: int,
    tree_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build a deterministic binary draft tree with the installed GPU kernel."""
    if tree_tokens == 7:
        topk, depth = 2, 3
        parent_values = [0, 0, 1, 2, 3]
    elif tree_tokens == 15:
        topk, depth = 2, 4
        parent_values = [0, 0, 1, 2, 3, 4, 5]
    else:
        raise ValueError(f"Unsupported tree_tokens={tree_tokens}; expected 7 or 15")

    parent_list = torch.tensor(
        [parent_values] * batch_size, dtype=torch.int64, device=device
    )
    selected_index = torch.arange(
        tree_tokens - 1, dtype=torch.int64, device=device
    ).repeat(batch_size, 1)
    verified_seq_len = torch.ones(batch_size, dtype=torch.int64, device=device)
    tree_mask = torch.zeros(
        batch_size * tree_tokens + batch_size * tree_tokens**2,
        dtype=torch.bool,
        device=device,
    )
    positions = torch.empty(
        batch_size * tree_tokens, dtype=torch.int64, device=device
    )
    retrive_index = torch.full(
        (batch_size, tree_tokens), -1, dtype=torch.int64, device=device
    )
    retrive_next_token = torch.full(
        (batch_size, tree_tokens), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (batch_size, tree_tokens), -1, dtype=torch.int64, device=device
    )

    build_tree_kernel_efficient(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        tree_tokens,
        0,
    )
    torch.cuda.synchronize()
    return (
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        positions,
    )


def make_workload(
    batch_size: int,
    tree_tokens: int,
    vocab_size: int,
    match_count: int,
    seed: int,
    device: torch.device,
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic target probabilities, predictions, and draft candidates."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    target_logits = torch.randn(
        (batch_size, tree_tokens, vocab_size),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    target_probs = torch.softmax(target_logits, dim=-1)
    target_predict = torch.argmax(target_probs, dim=-1).to(torch.int64)
    candidates = torch.empty_like(target_predict)
    candidates[:, 0] = target_predict[:, 0]
    for node in range(1, tree_tokens):
        parent = int(parent_indices[node].item())
        if int(depths[node].item()) <= match_count:
            candidates[:, node] = target_predict[:, parent]

    mismatch_generator = torch.Generator(device=device)
    mismatch_generator.manual_seed(seed + 1)
    random_tokens = torch.randint(
        0,
        vocab_size,
        (batch_size, tree_tokens),
        generator=mismatch_generator,
        dtype=torch.int64,
        device=device,
    )
    mismatch_mask = depths > match_count
    candidates[:, mismatch_mask] = torch.where(
        random_tokens[:, mismatch_mask] == target_predict[:, mismatch_mask],
        (random_tokens[:, mismatch_mask] + 1) % vocab_size,
        random_tokens[:, mismatch_mask],
    )
    return target_probs, target_predict, candidates


def allocate_outputs(
    batch_size: int,
    tree_tokens: int,
    depth: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicts = torch.full(
        (batch_size * tree_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (batch_size, depth), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)
    return predicts, accept_index, accept_token_num


def run_unchunked(
    workload: dict[str, torch.Tensor],
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    depth: int,
) -> None:
    predicts, accept_index, accept_token_num = outputs
    verify_tree_greedy(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=workload["candidates"],
        retrive_index=workload["retrive_index"],
        retrive_next_token=workload["retrive_next_token"],
        retrive_next_sibling=workload["retrive_next_sibling"],
        target_predict=workload["target_predict"],
    )


def prepare_chunk_specs(
    workload: dict[str, torch.Tensor], depth: int, chunk_count: int
) -> list[dict[str, torch.Tensor | int]]:
    batch_size = workload["candidates"].shape[0]
    tree_tokens = workload["candidates"].shape[1]
    chunk_size = batch_size // chunk_count
    specs = []
    for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        local_retrive_index = (
            workload["retrive_index"][start:stop] - start * tree_tokens
        )
        local_predicts = torch.empty(
            ((stop - start) * tree_tokens,),
            dtype=torch.int32,
            device=workload["candidates"].device,
        )
        local_accept_index = torch.empty(
            (stop - start, depth),
            dtype=torch.int32,
            device=workload["candidates"].device,
        )
        local_accept_token_num = torch.empty(
            stop - start, dtype=torch.int32, device=workload["candidates"].device
        )
        specs.append(
            {
                "start": start,
                "stop": stop,
                "local_retrive_index": local_retrive_index,
                "local_predicts": local_predicts,
                "local_accept_index": local_accept_index,
                "local_accept_token_num": local_accept_token_num,
                "candidates": workload["candidates"][start:stop],
                "retrive_next_token": workload["retrive_next_token"][start:stop],
                "retrive_next_sibling": workload["retrive_next_sibling"][start:stop],
                "target_predict": workload["target_predict"][start:stop],
            }
        )
    return specs


def reset_chunk_specs(specs: list[dict[str, torch.Tensor | int]]) -> None:
    for spec in specs:
        spec["local_predicts"].fill_(-1)
        spec["local_accept_index"].fill_(-1)
        spec["local_accept_token_num"].zero_()


def run_chunked_kernels(
    specs: list[dict[str, torch.Tensor | int]],
) -> None:
    for spec in specs:
        verify_tree_greedy(
            predicts=spec["local_predicts"],
            accept_index=spec["local_accept_index"],
            accept_token_num=spec["local_accept_token_num"],
            candidates=spec["candidates"],
            retrive_index=spec["local_retrive_index"],
            retrive_next_token=spec["retrive_next_token"],
            retrive_next_sibling=spec["retrive_next_sibling"],
            target_predict=spec["target_predict"],
        )


def map_chunked_outputs(
    workload: dict[str, torch.Tensor],
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    specs: list[dict[str, torch.Tensor | int]],
) -> None:
    global_predicts, global_accept_index, global_accept_token_num = outputs
    tree_tokens = workload["candidates"].shape[1]
    for spec in specs:
        start = spec["start"]
        stop = spec["stop"]
        global_predicts[
            workload["retrive_index"][start:stop].reshape(-1)
        ] = spec["local_predicts"]
        global_accept_index[start:stop] = torch.where(
            spec["local_accept_index"] != -1,
            spec["local_accept_index"] + start * tree_tokens,
            torch.full_like(spec["local_accept_index"], -1),
        )
        global_accept_token_num[start:stop] = spec["local_accept_token_num"]


def reference_greedy(
    workload: dict[str, torch.Tensor], depth: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent CPU reference for the greedy tree verifier."""
    candidates = workload["candidates"].cpu()
    retrive_index = workload["retrive_index"].cpu()
    retrive_next_token = workload["retrive_next_token"].cpu()
    retrive_next_sibling = workload["retrive_next_sibling"].cpu()
    target_predict = workload["target_predict"].cpu()
    batch_size, tree_tokens = candidates.shape
    predicts = torch.full(
        (batch_size * tree_tokens,), -1, dtype=torch.int32
    )
    accept_index = torch.full(
        (batch_size, depth), -1, dtype=torch.int32
    )
    accept_token_num = torch.zeros(batch_size, dtype=torch.int32)
    flat_target_predict = target_predict.reshape(-1)

    for row in range(batch_size):
        current_index = 0
        last_accepted_retrive_idx = int(retrive_index[row, current_index])
        accept_index[row, 0] = last_accepted_retrive_idx
        accepted_count = 0

        for _ in range(1, depth):
            current_index = int(retrive_next_token[row, current_index])
            while current_index != -1:
                draft_retrive_idx = int(retrive_index[row, current_index])
                draft_token = int(candidates[row, current_index])
                target_token = int(flat_target_predict[last_accepted_retrive_idx])
                if draft_token == target_token:
                    predicts[last_accepted_retrive_idx] = target_token
                    accepted_count += 1
                    accept_index[row, accepted_count] = draft_retrive_idx
                    last_accepted_retrive_idx = draft_retrive_idx
                    break
                current_index = int(retrive_next_sibling[row, current_index])
            if current_index == -1:
                break

        accept_token_num[row] = accepted_count
        predicts[last_accepted_retrive_idx] = int(
            flat_target_predict[last_accepted_retrive_idx]
        )
    return predicts, accept_index, accept_token_num


def reset_outputs(outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    predicts, accept_index, accept_token_num = outputs
    predicts.fill_(-1)
    accept_index.fill_(-1)
    accept_token_num.zero_()


def time_case(
    workload: dict[str, torch.Tensor],
    depth: int,
    chunk_count: int,
    warmup_count: int,
    measured_count: int,
) -> dict[str, Any]:
    batch_size = workload["candidates"].shape[0]
    tree_tokens = workload["candidates"].shape[1]
    outputs = allocate_outputs(batch_size, tree_tokens, depth, workload["candidates"].device)
    chunk_specs = (
        prepare_chunk_specs(workload, depth, chunk_count)
        if chunk_count > 1
        else None
    )

    for _ in range(warmup_count):
        reset_outputs(outputs)
        if chunk_count == 1:
            run_unchunked(workload, outputs, depth)
        else:
            reset_chunk_specs(chunk_specs)
            run_chunked_kernels(chunk_specs)
            map_chunked_outputs(workload, outputs, chunk_specs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_ms: list[float] = []
    for _ in range(measured_count):
        reset_outputs(outputs)
        torch.cuda.synchronize()
        start_event.record()
        if chunk_count == 1:
            run_unchunked(workload, outputs, depth)
        else:
            run_chunked_kernels(chunk_specs)
        end_event.record()
        torch.cuda.synchronize()
        if chunk_count > 1:
            map_chunked_outputs(workload, outputs, chunk_specs)
        elapsed_ms.append(start_event.elapsed_time(end_event))

    useful_tokens = int((outputs[0] != -1).sum().item())
    accepted_tokens = int(outputs[2].sum().item())
    median_ms = statistics.median(elapsed_ms)
    q1, q3 = statistics.quantiles(elapsed_ms, n=4)[0], statistics.quantiles(
        elapsed_ms, n=4
    )[2]
    return {
        "chunk_count": chunk_count,
        "chunk_rows": [batch_size // chunk_count] * chunk_count,
        "elapsed_ms_samples": elapsed_ms,
        "median_ms": median_ms,
        "mean_ms": statistics.mean(elapsed_ms),
        "stddev_ms": statistics.stdev(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
        "iqr_ms": q3 - q1,
        "useful_tokens": useful_tokens,
        "accepted_tokens": accepted_tokens,
        "useful_tokens_per_second": useful_tokens / (median_ms / 1000.0),
        "predicts": outputs[0].detach().cpu().clone(),
        "accept_index": outputs[1].detach().cpu().clone(),
        "accept_token_num": outputs[2].detach().cpu().clone(),
    }


def tensors_equal(
    left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]
) -> bool:
    return all(torch.equal(a, b) for a, b in zip(left, right))


def run_workload(
    name: str,
    batch_size: int,
    tree_tokens: int,
    vocab_size: int,
    match_count: int,
    chunk_counts: list[int],
    seed: int,
    device: torch.device,
    warmup_count: int,
    measured_count: int,
) -> dict[str, Any]:
    depth = 3 if tree_tokens == 7 else 4
    retrive_index, retrive_next_token, retrive_next_sibling, positions = build_tree(
        batch_size, tree_tokens, device
    )
    parent_indices = torch.full((tree_tokens,), -1, dtype=torch.int64, device=device)
    for parent in range(tree_tokens):
        child = int(retrive_next_token[0, parent].item())
        while child != -1:
            parent_indices[child] = parent
            child = int(retrive_next_sibling[0, child].item())
    depths = positions[:tree_tokens] - 1
    target_probs, target_predict, candidates = make_workload(
        batch_size,
        tree_tokens,
        vocab_size,
        match_count,
        seed,
        device,
        parent_indices,
        depths,
    )
    workload = {
        "retrive_index": retrive_index,
        "retrive_next_token": retrive_next_token,
        "retrive_next_sibling": retrive_next_sibling,
        "positions": positions,
        "target_probs": target_probs,
        "target_predict": target_predict,
        "candidates": candidates,
    }

    reference = reference_greedy(workload, depth)
    results: list[dict[str, Any]] = []
    for chunk_count in chunk_counts:
        torch.cuda.reset_peak_memory_stats()
        case = time_case(
            workload,
            depth,
            chunk_count,
            warmup_count,
            measured_count,
        )
        case["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        case["current_allocated_bytes"] = torch.cuda.memory_allocated()
        case["reserved_bytes"] = torch.cuda.memory_reserved()
        case["matches_reference"] = tensors_equal(
            (case["predicts"], case["accept_index"], case["accept_token_num"]),
            reference,
        )
        results.append(case)

    unchunked = results[0]
    for case in results[1:]:
        case["matches_unchunked"] = tensors_equal(
            (case["predicts"], case["accept_index"], case["accept_token_num"]),
            (
                unchunked["predicts"],
                unchunked["accept_index"],
                unchunked["accept_token_num"],
            ),
        )
        case["useful_work_matches_unchunked"] = (
            case["useful_tokens"] == unchunked["useful_tokens"]
        )

    for case in results:
        case.pop("predicts")
        case.pop("accept_index")
        case.pop("accept_token_num")

    return {
        "name": name,
        "batch_size": batch_size,
        "tree_tokens": tree_tokens,
        "vocab_size": vocab_size,
        "match_count": match_count,
        "depth": depth,
        "seed": seed,
        "dtypes": {
            "target_probs": "float32",
            "target_predict": "int64",
            "candidates": "int64",
            "retrive_index": "int64",
            "retrive_next_token": "int64",
            "retrive_next_sibling": "int64",
            "predicts": "int32",
            "accept_index": "int32",
            "accept_token_num": "int32",
        },
        "reference": {
            "type": "independent CPU greedy-tree traversal",
            "predicts_match": all(case["matches_reference"] for case in results),
            "accept_index_match": all(case["matches_reference"] for case in results),
            "accept_token_num_match": all(
                case["matches_reference"] for case in results
            ),
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-count", type=int, default=3)
    parser.add_argument("--measured-count", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one GPU is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    started = time.monotonic()
    small = run_workload(
        "small",
        batch_size=8,
        tree_tokens=7,
        vocab_size=128,
        match_count=3,
        chunk_counts=[1, 2, 4],
        seed=30344,
        device=device,
        warmup_count=args.warmup_count,
        measured_count=args.measured_count,
    )
    large = run_workload(
        "large",
        batch_size=2048,
        tree_tokens=15,
        vocab_size=2048,
        match_count=5,
        chunk_counts=[1, 8, 16],
        seed=30345,
        device=device,
        warmup_count=args.warmup_count,
        measured_count=args.measured_count,
    )
    elapsed = time.monotonic() - started

    properties = torch.cuda.get_device_properties(0)
    result = {
        "label": "installed-source gfx942 greedy verifier chunk-partition study",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "image": IMAGE_ID,
        "gpu": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "uuid": str(properties.uuid),
            "total_memory_bytes": properties.total_memory,
        },
        "torch": {
            "version": torch.__version__,
            "hip": torch.version.hip,
            "path": torch.__file__,
        },
        "sgl_kernel": {
            "python_path": __import__("sgl_kernel").__file__,
            "native_path": "/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so",
            "registered_ops_used": [
                "sgl_kernel::build_tree_kernel_efficient",
                "sgl_kernel::verify_tree_greedy",
            ],
            "unsupported_ops": [
                "sgl_kernel::tree_speculative_sampling_target_only",
                "sgl_kernel::top_k_renorm_probs",
            ],
        },
        "commands": {
            "benchmark": [
                "/opt/venv/bin/python",
                "reports/j-df14d49951a3/chunk_partition.py",
                "--output",
                "reports/j-df14d49951a3/results.json",
            ],
            "installed_baseline": [
                "/opt/venv/bin/python",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "/sgl-workspace/sglang/python/sglang/kernels/aot/tests/speculative/test_eagle_utils.py",
            ],
        },
        "timing_method": (
            f"{args.warmup_count} warmups and {args.measured_count} measured "
            "iterations per case using CUDA events; outputs reset and chunk-result "
            "mapping occur outside timing"
        ),
        "numerical_gates": {
            "independent_reference": "exact equality with independent CPU greedy-tree traversal",
            "chunked_vs_unchunked": "exact equality for predicts, accept_index, and accept_token_num",
            "useful_work": "exact equality of non--1 predicts count",
        },
        "workloads": [small, large],
        "elapsed_seconds": elapsed,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
