#!/usr/bin/env python3
"""Bounded one-GPU DSpark reduced-pipeline study on gfx942.

This study uses synthetic weights and states.  It deliberately stops before any
model-specific GLM constructor: the upstream issue's full path requires a real
target/draft checkpoint and model configuration.  The reduced boundary still
exercises production SGLang primitives and compares every handoff to an
independent reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import triton

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    FinalizeAcceptLens,
    accept_greedy,
    accept_greedy_triton,
)
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import CommitKvProj
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens
from sglang.srt.models.dspark import project_through_lm_head
from sglang.srt.speculative.dspark_components.dspark_draft import (
    select_draft_hidden_without_anchor,
)


@dataclass(frozen=True)
class Case:
    prefill_tokens: int
    spatial_batch: int


class LinearStub(torch.nn.Module):
    quant_method = None

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(hidden, self.weight), None


_RETAINED_COMMIT_MODULES: list[list[torch.nn.Module]] = []


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def module_weight_bytes(modules: list[torch.nn.Module]) -> int:
    return sum(tensor_bytes(value) for module in modules for value in module.parameters())


def max_abs_error_chunked(
    actual: torch.Tensor, weight: torch.Tensor, hidden: torch.Tensor, chunk: int = 2048
) -> float:
    max_error = 0.0
    for start in range(0, hidden.shape[0], chunk):
        stop = min(start + chunk, hidden.shape[0])
        reference = torch.matmul(
            hidden[start:stop].float(), weight.float().transpose(0, 1)
        )
        max_error = max(
            float((actual[start:stop].float() - reference).abs().max()), max_error
        )
    return max_error


def event_seconds(forward) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = forward()
    end.record()
    torch.cuda.synchronize()
    # Keep the result alive until synchronization completes.
    del result
    return float(start.elapsed_time(end) / 1000.0)


def run_case(
    case: Case,
    *,
    device: torch.device,
    seed: int,
    warm_forwards: int,
) -> dict[str, Any]:
    if case.prefill_tokens < case.spatial_batch * 6:
        raise ValueError("prefill_tokens must cover the six-token verify window")

    generator = torch.Generator(device=device).manual_seed(seed)
    target_hidden_size = 4096
    draft_hidden_size = 1024
    vocab_size = 8192
    gamma = 5
    stride = gamma + 1
    commit_stages = 3
    commit_head_dim = 576
    dtype = torch.bfloat16

    target_hidden = (
        torch.randn(
            (case.prefill_tokens, target_hidden_size),
            device=device,
            generator=generator,
        )
        * 0.25
    ).to(dtype)
    lm_head_weight = (
        torch.randn((vocab_size, target_hidden_size), device=device, generator=generator)
        * 0.01
    ).to(dtype)
    lm_head = LinearStub(lm_head_weight).to(device)

    draft_rows = case.spatial_batch * stride
    draft_hidden = (
        torch.randn((draft_rows, draft_hidden_size), device=device, generator=generator)
        * 0.25
    ).to(dtype)
    commit_weights = [
        (
            torch.randn(
                (commit_head_dim, draft_hidden_size), device=device, generator=generator
            )
            * 0.02
        ).to(dtype)
        for _ in range(commit_stages)
    ]
    commit_linears = [LinearStub(weight).to(device) for weight in commit_weights]
    _RETAINED_COMMIT_MODULES.append(commit_linears)

    draft_tokens = torch.randint(
        0, vocab_size, (case.spatial_batch, gamma), device=device, generator=generator
    )
    verify_lens = torch.randint(
        1, stride + 1, (case.spatial_batch,), device=device, generator=generator
    ).to(torch.int32)
    prefix_lens = torch.randint(
        1, 4096, (case.spatial_batch,), device=device, generator=generator
    ).to(torch.int64)
    reference_verify_logits = torch.matmul(
        target_hidden[:draft_rows].float(), lm_head_weight.float().transpose(0, 1)
    )
    target_predictions = reference_verify_logits.argmax(dim=-1).view(
        case.spatial_batch, stride
    )
    candidates = torch.randint(
        0, vocab_size, (case.spatial_batch, stride), device=device, generator=generator
    )
    intended_accept_lens = torch.randint(
        0, gamma, (case.spatial_batch,), device=device, generator=generator
    ).to(torch.int64)
    for row, accept_len in enumerate(intended_accept_lens.tolist()):
        candidates[row, 1 : accept_len + 1] = target_predictions[row, :accept_len]
        if accept_len < gamma:
            mismatch = (target_predictions[row, accept_len] + 1) % vocab_size
            candidates[row, accept_len + 1] = mismatch
    del reference_verify_logits, target_predictions, intended_accept_lens

    def forward() -> tuple[Any, ...]:
        target_logits = project_through_lm_head(target_hidden, lm_head)
        selected_draft_hidden, selected_draft_hidden_3d = (
            select_draft_hidden_without_anchor(
                draft_hidden, bs=case.spatial_batch, gamma=gamma
            )
        )
        committed_kv = CommitKvProj.triton(
            main_x=selected_draft_hidden, wkv_linears=commit_linears
        )
        correct_len, bonus, cap_trim_lens = accept_greedy_triton(
            candidates=candidates,
            target_logits=target_logits[:draft_rows],
            verify_num_draft_tokens=stride,
            cutoff_verify_lens=verify_lens,
        )
        finalized = FinalizeAcceptLens.triton(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )
        out_tokens = BuildOutTokens.triton(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=stride,
            gamma=gamma,
        )
        return (
            target_logits,
            selected_draft_hidden,
            selected_draft_hidden_3d,
            committed_kv,
            correct_len,
            bonus,
            cap_trim_lens,
            finalized,
            out_tokens,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cold_start = time.perf_counter()
    cold_seconds = event_seconds(forward)
    cold_wall_seconds = time.perf_counter() - cold_start
    cold = forward()
    torch.cuda.synchronize()

    (
        target_logits,
        selected_draft_hidden,
        selected_draft_hidden_3d,
        committed_kv,
        correct_len,
        bonus,
        cap_trim_lens,
        finalized,
        out_tokens,
    ) = cold
    projection_error = max_abs_error_chunked(
        actual=target_logits, weight=lm_head_weight, hidden=target_hidden
    )
    expected_selected, expected_selected_3d = select_draft_hidden_without_anchor(
        draft_hidden, bs=case.spatial_batch, gamma=gamma
    )
    draft_selection_exact = bool(
        torch.equal(selected_draft_hidden, expected_selected)
        and torch.equal(selected_draft_hidden_3d, expected_selected_3d)
    )
    expected_correct, expected_bonus, expected_trim = accept_greedy(
        candidates=candidates,
        target_logits=target_logits[:draft_rows],
        verify_num_draft_tokens=stride,
        cutoff_verify_lens=verify_lens,
    )
    accept_exact = bool(
        torch.equal(correct_len, expected_correct)
        and torch.equal(bonus, expected_bonus)
        and torch.equal(cap_trim_lens, expected_trim)
    )
    expected_final = FinalizeAcceptLens.torch(
        correct_len=correct_len,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=prefix_lens,
    )
    finalize_exact = bool(
        torch.equal(finalized.commit_lens, expected_final.commit_lens)
        and torch.equal(finalized.new_seq_lens, expected_final.new_seq_lens)
        and torch.equal(finalized.cap_trim_lens, expected_final.cap_trim_lens)
    )
    expected_out_tokens = BuildOutTokens.torch(
        draft_tokens=draft_tokens,
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=stride,
        gamma=gamma,
    )
    out_tokens_exact = bool(torch.equal(out_tokens, expected_out_tokens))
    expected_committed_kv = CommitKvProj.torch(
        main_x=selected_draft_hidden, wkv_linears=commit_linears
    )
    commit_errors = [
        float((got.float() - ref.float()).abs().max())
        for got, ref in zip(committed_kv, expected_committed_kv)
    ]
    commit_projection_error = max(commit_errors, default=0.0)

    warm_seconds = [event_seconds(forward) for _ in range(warm_forwards)]
    median_warm_seconds = statistics.median(warm_seconds)
    accepted_tokens = int(finalized.commit_lens.sum().item())
    weight_bytes = module_weight_bytes([lm_head, *commit_linears])
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    current_allocated_bytes = int(torch.cuda.memory_allocated())

    gates = {
        "target_projection_max_abs_le_0.02": projection_error <= 0.02,
        "draft_hidden_selection_exact": draft_selection_exact,
        "accept_greedy_exact": accept_exact,
        "finalize_accept_exact": finalize_exact,
        "build_out_tokens_exact": out_tokens_exact,
        "commit_projection_max_abs_le_0.02": commit_projection_error <= 0.02,
        "weights_under_4gb": weight_bytes < 4 * 1024**3,
        "peak_allocations_under_48gb": peak_allocated_bytes < 48 * 1024**3,
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"accuracy or resource gate failed: {gates}; "
            f"projection_error={projection_error:.9g}, "
            f"commit_projection_error={commit_projection_error:.9g}, "
            f"commit_stage_errors={[f'{value:.9g}' for value in commit_errors]}"
        )

    result = {
        "case": {
            "prefill_tokens": case.prefill_tokens,
            "spatial_batch": case.spatial_batch,
            "draft_rows": draft_rows,
            "gamma": gamma,
            "verify_num_draft_tokens": stride,
        },
        "dimensions": {
            "target_hidden": [case.prefill_tokens, target_hidden_size],
            "target_logits": [case.prefill_tokens, vocab_size],
            "draft_hidden": [draft_rows, draft_hidden_size],
            "selected_draft_hidden": [case.spatial_batch * gamma, draft_hidden_size],
            "commit_kv_per_stage": [case.spatial_batch * gamma, commit_head_dim],
            "verify_logits": [draft_rows, vocab_size],
        },
        "dtypes": {
            "hidden_states": str(dtype),
            "weights": str(dtype),
            "logits": str(dtype),
            "token_ids": "torch.int64",
            "verify_lens": "torch.int32",
            "prefix_lens": "torch.int64",
        },
        "accuracy": {
            "target_projection_max_abs_error": projection_error,
            "commit_projection_max_abs_error": commit_projection_error,
            "draft_hidden_selection_exact": draft_selection_exact,
            "accept_greedy_exact": accept_exact,
            "finalize_accept_exact": finalize_exact,
            "build_out_tokens_exact": out_tokens_exact,
            "accepted_tokens": accepted_tokens,
        },
        "gates": gates,
        "timing": {
            "cold_cuda_event_seconds": cold_seconds,
            "cold_wall_seconds": cold_wall_seconds,
            "warm_cuda_event_seconds": warm_seconds,
            "warm_median_seconds": median_warm_seconds,
            "prefill_tokens_per_second": case.prefill_tokens / median_warm_seconds,
            "committed_draft_tokens_per_second": case.spatial_batch
            * gamma
            / median_warm_seconds,
            "pipeline_tokens_per_second": (
                case.prefill_tokens + case.spatial_batch * gamma
            )
            / median_warm_seconds,
            "accepted_tokens_per_second": accepted_tokens / median_warm_seconds,
        },
        "resources": {
            "weight_bytes": weight_bytes,
            "peak_allocated_bytes": peak_allocated_bytes,
            "current_allocated_bytes": current_allocated_bytes,
        },
    }

    del cold, target_logits, selected_draft_hidden, selected_draft_hidden_3d
    del committed_kv, finalized, out_tokens, expected_committed_kv
    del target_hidden, draft_hidden, lm_head, commit_linears
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-forwards", type=int, default=5)
    parser.add_argument("--seed", type=int, default=399)
    args = parser.parse_args()
    if not 1 <= args.warm_forwards <= 20:
        raise ValueError("--warm-forwards must be in [1, 20]")

    if not torch.cuda.is_available():
        raise RuntimeError("This study requires one available CUDA/HIP GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one GPU, observed {torch.cuda.device_count()}"
        )

    device = torch.device("cuda")
    cases = [
        Case(4096, 8),
        Case(8192, 16),
        Case(16384, 32),
        Case(24576, 64),
        Case(32768, 128),
        Case(32768, 256),
    ]
    started = time.perf_counter()
    results = [
        run_case(
            case,
            device=device,
            seed=args.seed + index,
            warm_forwards=args.warm_forwards,
        )
        for index, case in enumerate(cases)
    ]
    elapsed_seconds = time.perf_counter() - started

    report = {
        "schema_version": 1,
        "label": "synthetic reduced DSpark pipeline; no GLM weights or distributed claim",
        "model_specific_boundary": (
            "The model-specific GLM/DSpark target-hidden projector constructor was "
            "not instantiated because it requires a full model configuration and "
            "checkpoint. The reduced boundary starts from synthetic target hidden "
            "states and uses the generic production projection, draft-hidden, "
            "accept, finalize, output-commit, and commit-KV primitives."
        ),
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "torch_path": torch.__file__,
            "triton_path": triton.__file__,
            "sglang_path": __import__("sglang").__file__,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": torch.cuda.get_device_capability(device),
            "gpu_count": torch.cuda.device_count(),
        },
        "limits": {
            "wall_limit_seconds": 7200,
            "max_cases": 6,
            "max_weight_bytes": 4 * 1024**3,
            "max_live_allocated_bytes": 48 * 1024**3,
            "warm_forwards_per_case": args.warm_forwards,
            "cold_forwards_per_case": 2,
        },
        "timing_method": (
            "CUDA events around one complete reduced forward; cold event is first "
            "kernel-compiling call, warm events are the next bounded real forwards; "
            "median is reported"
        ),
        "cases": results,
        "total_study_seconds": elapsed_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
