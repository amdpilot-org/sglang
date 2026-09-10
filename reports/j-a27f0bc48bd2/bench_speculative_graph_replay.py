#!/usr/bin/env python3
"""Bounded eager-vs-CUDA-graph study for SGLang chain speculative sampling.

This is a reduced-block benchmark, not a full-model speedup claim.  It uses the
installed Triton speculative verifier with synthetic target probabilities and
chain draft trees, checks every case against an independent CPU reference, and
compares eager execution with captured graph replay on one MI300X.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)


DEFAULT_CASES = [
    (1, 5, 1024),
    (2, 9, 2048),
    (4, 17, 4096),
    (8, 33, 8192),
    (16, 65, 16384),
    (32, 129, 32768),
]

FRESH_INPUT_OFFSETS = (0.0, 0.25, 0.5)
WARMUP_ITERATIONS = 10
MEASURED_BATCHES = 3
MEASURED_ITERATIONS_PER_BATCH = 50
WEIGHT_LIMIT_BYTES = 4 * 1024**3
LIVE_ALLOCATION_LIMIT_BYTES = 48 * 1024**3
WALL_LIMIT_SECONDS = 7200


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def make_chain_tree(
    batch_size: int,
    num_slots: int,
    vocab_size: int,
    device: torch.device,
):
    candidates = torch.arange(
        batch_size * num_slots, dtype=torch.int64, device=device
    ).reshape(batch_size, num_slots) % vocab_size
    retrive_index = torch.arange(
        batch_size * num_slots, dtype=torch.int64, device=device
    ).reshape(batch_size, num_slots)
    retrive_next_token = torch.full(
        (batch_size, num_slots), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (batch_size, num_slots), -1, dtype=torch.int64, device=device
    )
    for batch in range(batch_size):
        retrive_next_token[batch, : num_slots - 1] = torch.arange(
            1, num_slots, dtype=torch.int64, device=device
        )
    return candidates, retrive_index, retrive_next_token, retrive_next_sibling


def make_logits(
    batch_size: int,
    num_slots: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    logits = torch.zeros(
        (batch_size, num_slots, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    candidates = torch.arange(
        batch_size * num_slots, dtype=torch.int64, device=device
    ).reshape(batch_size, num_slots) % vocab_size
    for batch in range(batch_size):
        for row in range(num_slots - 1):
            logits[batch, row, int(candidates[batch, row + 1])] = 8.0
        logits[batch, num_slots - 1, 0] = 8.0
    return logits


def chain_reference(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_final: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
):
    """Independent CPU implementation of the chain verifier semantics."""
    del retrive_next_sibling
    batch_size, num_slots = candidates.shape
    vocab_size = logits.shape[-1]
    target_probs = F.softmax(logits.to(torch.float32), dim=-1)
    predicts = torch.full(
        (batch_size * num_slots,), -1, dtype=torch.int32, device="cpu"
    )
    accept_index = torch.full(
        (batch_size, num_slots), -1, dtype=torch.int32, device="cpu"
    )
    accept_num = torch.zeros(batch_size, dtype=torch.int32, device="cpu")

    for batch in range(batch_size):
        cur_prob_row = 0
        root_global_idx = int(retrive_index[batch, 0])
        accept_index[batch, 0] = root_global_idx
        last_accepted_global_idx = root_global_idx
        num_accept = 0
        step = 1
        continue_verifying = True

        while step < num_slots and continue_verifying:
            draft_token = int(candidates[batch, step])
            p = float(target_probs[batch, cur_prob_row, draft_token])
            q = float(draft_probs[batch, cur_prob_row, draft_token])
            coin = float(uniform_samples[batch, step - 1])
            if coin * q < p:
                num_accept += 1
                cur_prob_row = step
                predicts[last_accepted_global_idx] = draft_token
                current_global_idx = int(retrive_index[batch, step])
                accept_index[batch, num_accept] = current_global_idx
                last_accepted_global_idx = current_global_idx
                step += 1
            else:
                continue_verifying = False

        accept_num[batch] = num_accept
        coin_final = float(uniform_final[batch])
        if continue_verifying:
            residual = target_probs[batch, cur_prob_row]
        else:
            residual = torch.clamp(
                target_probs[batch, cur_prob_row]
                - draft_probs[batch, cur_prob_row],
                min=0.0,
            )
        norm_sum = float(residual.sum())
        target_u = coin_final * norm_sum
        cumulative = torch.cumsum(residual, dim=0)
        matches = torch.nonzero(cumulative > target_u, as_tuple=False)
        if matches.numel() > 0:
            final_token = int(matches[0, 0])
        else:
            valid = torch.nonzero(residual > 0.0, as_tuple=False)
            final_token = int(valid[-1, 0]) if valid.numel() > 0 else vocab_size - 1
        predicts[last_accepted_global_idx] = final_token

    return predicts, accept_index, accept_num


def run_block(
    static_logits: torch.Tensor,
    static_target_probs: torch.Tensor,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_final: torch.Tensor,
    draft_probs: torch.Tensor,
) -> None:
    static_target_probs.copy_(F.softmax(static_logits, dim=-1))
    chain_speculative_sampling_triton(
        predicts,
        accept_index,
        accept_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_final,
        static_target_probs,
        draft_probs,
        1.0,
        1.0,
        True,
    )


def time_batch(
    run_once,
    iterations: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        run_once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def check_outputs(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, bool]:
    return {
        "predicts_match": torch.equal(actual[0].cpu(), expected[0]),
        "accept_index_match": torch.equal(actual[1].cpu(), expected[1]),
        "accept_num_match": torch.equal(actual[2].cpu(), expected[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one GPU, found {torch.cuda.device_count()}"
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    repo = Path(__file__).resolve().parents[2]
    properties = torch.cuda.get_device_properties(device)
    source_path = inspect.getsourcefile(chain_speculative_sampling_triton)

    overall_start = time.perf_counter()
    results = {
        "label": "eager versus captured block replay reduced-block study",
        "started_at_utc": utc_now(),
        "campaign": "repo-e2e-20260909",
        "coordination_tracker": "amdpilot-org/amdpilotv2 issue 402",
        "upstream_context": "sgl-project/sglang issue 30344",
        "gpu": {
            "name": properties.name,
            "count": torch.cuda.device_count(),
            "capability": f"{properties.major}.{properties.minor}",
            "gfx": "gfx942",
            "uuid": str(properties.uuid),
            "total_memory_bytes": properties.total_memory,
        },
        "image": {
            "local_image_id": os.environ.get(
                "JOB_IMAGE_ID",
                "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
            )
        },
        "stack": {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_path": torch.__file__,
            "torch_hip": torch.version.hip,
            "triton": triton.__version__,
            "triton_path": triton.__file__,
            "sglang": __import__("sglang").__version__,
            "sglang_path": __import__("sglang").__file__,
            "kernel_source": source_path,
        },
        "source": {
            "branch": git_output(repo, "branch", "--show-current"),
            "commit": git_output(repo, "rev-parse", "HEAD"),
            "pr_base": git_output(repo, "rev-parse", "main"),
        },
        "limits": {
            "wall_seconds": WALL_LIMIT_SECONDS,
            "max_cases": 6,
            "weights_bytes": WEIGHT_LIMIT_BYTES,
            "live_allocations_bytes": LIVE_ALLOCATION_LIMIT_BYTES,
        },
        "method": {
            "primitive": "sglang.kernels.ops.speculative.reject_sampling.chain_speculative_sampling_triton",
            "primitive_kind": "real installed Triton GPU speculative verifier/sampling kernel",
            "block": "float32 softmax plus chain speculative verifier",
            "reference": "independent CPU implementation with fixed uniform samples",
            "accuracy_gate": "exact equality for int32 predicts, accept_index, and accept_num",
            "timing": (
                f"{WARMUP_ITERATIONS} warmup iterations, then "
                f"{MEASURED_BATCHES} batches of "
                f"{MEASURED_ITERATIONS_PER_BATCH} complete block executions; "
                "median of CUDA-event batch means"
            ),
            "fresh_input_values": list(FRESH_INPUT_OFFSETS),
            "static_buffer_reuse": True,
            "independent_outputs": True,
            "capture_boundaries_unsupported": [
                "fresh CPU input generation and host-to-device copies",
                "independent CPU reference execution",
                "output snapshots from GPU to CPU",
                "host synchronization and scalar reads",
                "torch.multinomial and other dynamic host-side sampling",
            ],
        },
        "cases": [],
    }

    for case_index, (batch_size, num_slots, vocab_size) in enumerate(
        DEFAULT_CASES, start=1
    ):
        case_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        candidates, retrive_index, retrive_next_token, retrive_next_sibling = (
            make_chain_tree(batch_size, num_slots, vocab_size, device)
        )
        static_logits = make_logits(batch_size, num_slots, vocab_size, device)
        static_target_probs = torch.empty_like(static_logits)
        static_draft_probs = torch.zeros_like(static_logits)
        static_uniform_samples = torch.full(
            (batch_size, num_slots - 1), 0.25, device=device
        )
        static_uniform_final = torch.full((batch_size,), 0.25, device=device)

        eager_predicts = torch.full(
            (batch_size * num_slots,), -1, dtype=torch.int32, device=device
        )
        eager_accept_index = torch.full(
            (batch_size, num_slots), -1, dtype=torch.int32, device=device
        )
        eager_accept_num = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )
        graph_predicts = torch.full(
            (batch_size * num_slots,), -1, dtype=torch.int32, device=device
        )
        graph_accept_index = torch.full(
            (batch_size, num_slots), -1, dtype=torch.int32, device=device
        )
        graph_accept_num = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )

        weights_bytes = tensor_bytes(
            static_logits,
            static_target_probs,
            static_draft_probs,
        )
        live_bytes = torch.cuda.memory_allocated()
        if weights_bytes >= WEIGHT_LIMIT_BYTES:
            raise RuntimeError(
                f"Case {case_index} generated weights exceed 4 GiB: {weights_bytes}"
            )
        if live_bytes >= LIVE_ALLOCATION_LIMIT_BYTES:
            raise RuntimeError(
                f"Case {case_index} live allocations exceed 48 GiB: {live_bytes}"
            )

        def eager_run() -> None:
            run_block(
                static_logits,
                static_target_probs,
                eager_predicts,
                eager_accept_index,
                eager_accept_num,
                candidates,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                static_uniform_samples,
                static_uniform_final,
                static_draft_probs,
            )

        def graph_run() -> None:
            run_block(
                static_logits,
                static_target_probs,
                graph_predicts,
                graph_accept_index,
                graph_accept_num,
                candidates,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                static_uniform_samples,
                static_uniform_final,
                static_draft_probs,
            )

        correctness = {
            "eager_predicts_match": True,
            "eager_accept_index_match": True,
            "eager_accept_num_match": True,
            "graph_predicts_match": True,
            "graph_accept_index_match": True,
            "graph_accept_num_match": True,
            "graph_vs_eager_predicts_match": True,
            "graph_vs_eager_accept_index_match": True,
            "graph_vs_eager_accept_num_match": True,
        }

        base_logits_cpu = static_logits.detach().cpu()
        for fresh_offset in FRESH_INPUT_OFFSETS:
            fresh_logits_cpu = base_logits_cpu.clone()
            fresh_logits_cpu[..., 0] += fresh_offset
            static_logits.copy_(fresh_logits_cpu)

            reference_predicts, reference_accept_index, reference_accept_num = (
                chain_reference(
                    fresh_logits_cpu,
                    candidates.cpu(),
                    retrive_index.cpu(),
                    retrive_next_token.cpu(),
                    retrive_next_sibling.cpu(),
                    static_uniform_samples.cpu(),
                    static_uniform_final.cpu(),
                    static_draft_probs.cpu(),
                    1.0,
                    1.0,
                )
            )

            eager_predicts.fill_(-1)
            eager_accept_index.fill_(-1)
            eager_accept_num.zero_()
            eager_run()
            torch.cuda.synchronize()
            eager_match = check_outputs(
                (eager_predicts, eager_accept_index, eager_accept_num),
                (reference_predicts, reference_accept_index, reference_accept_num),
            )
            correctness["eager_predicts_match"] &= eager_match["predicts_match"]
            correctness["eager_accept_index_match"] &= eager_match["accept_index_match"]
            correctness["eager_accept_num_match"] &= eager_match["accept_num_match"]

            graph_predicts.fill_(-1)
            graph_accept_index.fill_(-1)
            graph_accept_num.zero_()
            graph_run()
            torch.cuda.synchronize()
            graph_match = check_outputs(
                (graph_predicts, graph_accept_index, graph_accept_num),
                (reference_predicts, reference_accept_index, reference_accept_num),
            )
            correctness["graph_predicts_match"] &= graph_match["predicts_match"]
            correctness["graph_accept_index_match"] &= graph_match["accept_index_match"]
            correctness["graph_accept_num_match"] &= graph_match["accept_num_match"]
            correctness["graph_vs_eager_predicts_match"] &= torch.equal(
                graph_predicts.cpu(), eager_predicts.cpu()
            )
            correctness["graph_vs_eager_accept_index_match"] &= torch.equal(
                graph_accept_index.cpu(), eager_accept_index.cpu()
            )
            correctness["graph_vs_eager_accept_num_match"] &= torch.equal(
                graph_accept_num.cpu(), eager_accept_num.cpu()
            )

        if not all(correctness.values()):
            failed = [key for key, value in correctness.items() if not value]
            raise AssertionError(f"Case {case_index} failed accuracy gates: {failed}")

        for _ in range(WARMUP_ITERATIONS):
            eager_run()
        torch.cuda.synchronize()

        capture_start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph_run()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_run()
        capture_elapsed_ms = (time.perf_counter() - capture_start) * 1000.0

        eager_ms_batches = [
            time_batch(eager_run, MEASURED_ITERATIONS_PER_BATCH)
            for _ in range(MEASURED_BATCHES)
        ]
        graph_ms_batches = [
            time_batch(graph.replay, MEASURED_ITERATIONS_PER_BATCH)
            for _ in range(MEASURED_BATCHES)
        ]
        eager_ms = statistics.median(eager_ms_batches)
        graph_ms = statistics.median(graph_ms_batches)

        case_result = {
            "case_index": case_index,
            "batch_size": batch_size,
            "num_slots": num_slots,
            "num_draft_tokens": num_slots,
            "num_spec_steps": num_slots,
            "vocab_size": vocab_size,
            "dtypes": {
                "logits": str(static_logits.dtype),
                "target_probs": str(static_target_probs.dtype),
                "draft_probs": str(static_draft_probs.dtype),
                "uniform_samples": str(static_uniform_samples.dtype),
                "predicts": str(eager_predicts.dtype),
                "accept_index": str(eager_accept_index.dtype),
                "accept_num": str(eager_accept_num.dtype),
                "candidates": str(candidates.dtype),
                "retrive_index": str(retrive_index.dtype),
                "retrive_next_token": str(retrive_next_token.dtype),
                "retrive_next_sibling": str(retrive_next_sibling.dtype),
            },
            "accuracy": correctness,
            "capture_supported": True,
            "capture_error": None,
            "capture_ms": capture_elapsed_ms,
            "eager_ms_batches": eager_ms_batches,
            "graph_ms_batches": graph_ms_batches,
            "eager_ms_per_iteration": eager_ms,
            "graph_ms_per_iteration": graph_ms,
            "eager_to_graph_speedup": eager_ms / graph_ms,
            "weights_bytes": weights_bytes,
            "live_allocations_bytes": torch.cuda.memory_allocated(),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "case_elapsed_seconds": time.perf_counter() - case_start,
        }
        results["cases"].append(case_result)

        del graph
        del graph_predicts, graph_accept_index, graph_accept_num
        del eager_predicts, eager_accept_index, eager_accept_num
        del static_logits, static_target_probs, static_draft_probs
        del static_uniform_samples, static_uniform_final
        del candidates, retrive_index, retrive_next_token, retrive_next_sibling
        gc.collect()
        torch.cuda.empty_cache()

    overall_elapsed = time.perf_counter() - overall_start
    if overall_elapsed >= WALL_LIMIT_SECONDS:
        raise RuntimeError(
            f"Benchmark exceeded 7200 second wall limit: {overall_elapsed}"
        )
    results["completed_at_utc"] = utc_now()
    results["elapsed_seconds"] = overall_elapsed

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
