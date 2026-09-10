#!/usr/bin/env python3
"""Bounded gfx942 DSpark chunk-partition study.

The workload uses one seeded synthetic target/draft state and six legal row
partitions.  It compares each partition with both an independent reference and
the unchunked production-primitive result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    FinalizeAcceptLens,
    accept_greedy_triton,
)
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import CommitKvProj
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens
from sglang.srt.models.dspark import project_through_lm_head
from sglang.srt.speculative.dspark_components.dspark_draft import (
    select_draft_hidden_without_anchor,
)


PREFILL_TOKENS = 32768
SPATIAL_BATCH = 128
TARGET_HIDDEN_SIZE = 4096
DRAFT_HIDDEN_SIZE = 1024
VOCAB_SIZE = 8192
GAMMA = 5
STRIDE = GAMMA + 1
COMMIT_STAGES = 3
COMMIT_HEAD_DIM = 576
DTYPE = torch.bfloat16
SEED = 30734
WARM_FORWARDS = 5
WEIGHT_LIMIT_BYTES = 4 * 1024**3
PEAK_ALLOCATION_LIMIT_BYTES = 48 * 1024**3
PROJECTION_MAX_ABS_GATE = 0.02


@dataclass(frozen=True)
class PartitionCase:
    name: str
    target_chunk: int
    commit_chunk: int


CASES = (
    PartitionCase("full_full", PREFILL_TOKENS, SPATIAL_BATCH * GAMMA),
    PartitionCase("half_full", PREFILL_TOKENS // 2, SPATIAL_BATCH * GAMMA),
    PartitionCase("full_half", PREFILL_TOKENS, SPATIAL_BATCH * GAMMA // 2),
    PartitionCase("half_half", PREFILL_TOKENS // 2, SPATIAL_BATCH * GAMMA // 2),
    PartitionCase("quarter_quarter", PREFILL_TOKENS // 4, SPATIAL_BATCH * GAMMA // 4),
    PartitionCase("eighth_eighth", PREFILL_TOKENS // 8, SPATIAL_BATCH * GAMMA // 8),
)


class LinearStub(torch.nn.Module):
    quant_method = None

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(hidden, self.weight), None


@dataclass
class PipelineResult:
    target_logits: torch.Tensor
    selected_draft_hidden: torch.Tensor
    selected_draft_hidden_3d: torch.Tensor
    committed_kv: list[torch.Tensor]
    correct_len: torch.Tensor
    bonus: torch.Tensor
    cap_trim_lens: torch.Tensor
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    out_tokens: torch.Tensor


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def module_weight_bytes(modules: list[torch.nn.Module]) -> int:
    return sum(
        tensor_bytes(value)
        for module in modules
        for value in module.parameters()
    )


def chunk_bounds(rows: int, chunk: int) -> list[tuple[int, int]]:
    if rows <= 0 or chunk <= 0 or rows % chunk != 0:
        raise ValueError(f"illegal partition: rows={rows}, chunk={chunk}")
    return [(start, start + chunk) for start in range(0, rows, chunk)]


def project_chunked(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    chunk: int,
) -> torch.Tensor:
    if chunk == hidden.shape[0]:
        return project_through_lm_head(hidden, lm_head)
    output = torch.empty(
        (hidden.shape[0], lm_head.weight.shape[0]),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    for start, stop in chunk_bounds(hidden.shape[0], chunk):
        output[start:stop] = project_through_lm_head(hidden[start:stop], lm_head)
    return output


def commit_chunked(
    hidden: torch.Tensor,
    linears: list[torch.nn.Module],
    chunk: int,
) -> list[torch.Tensor]:
    if chunk == hidden.shape[0]:
        return CommitKvProj.triton(main_x=hidden, wkv_linears=linears)
    outputs = [
        torch.empty(
            (hidden.shape[0], linear.weight.shape[0]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        for linear in linears
    ]
    for start, stop in chunk_bounds(hidden.shape[0], chunk):
        chunk_outputs = CommitKvProj.triton(
            main_x=hidden[start:stop], wkv_linears=linears
        )
        for output, chunk_output in zip(outputs, chunk_outputs):
            output[start:stop].copy_(chunk_output)
    return outputs


def run_pipeline(
    *,
    target_hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    draft_hidden: torch.Tensor,
    commit_linears: list[torch.nn.Module],
    candidates: torch.Tensor,
    draft_tokens: torch.Tensor,
    verify_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    target_chunk: int,
    commit_chunk: int,
) -> PipelineResult:
    target_logits = project_chunked(target_hidden, lm_head, target_chunk)
    selected_draft_hidden, selected_draft_hidden_3d = (
        select_draft_hidden_without_anchor(
            draft_hidden, bs=SPATIAL_BATCH, gamma=GAMMA
        )
    )
    committed_kv = commit_chunked(
        selected_draft_hidden, commit_linears, commit_chunk
    )
    correct_len, bonus, cap_trim_lens = accept_greedy_triton(
        candidates=candidates,
        target_logits=target_logits[: SPATIAL_BATCH * STRIDE],
        verify_num_draft_tokens=STRIDE,
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
        verify_num_draft_tokens=STRIDE,
        gamma=GAMMA,
    )
    return PipelineResult(
        target_logits=target_logits,
        selected_draft_hidden=selected_draft_hidden,
        selected_draft_hidden_3d=selected_draft_hidden_3d,
        committed_kv=committed_kv,
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=cap_trim_lens,
        commit_lens=finalized.commit_lens,
        new_seq_lens=finalized.new_seq_lens,
        out_tokens=out_tokens,
    )


def max_abs_error_fp32(
    actual: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> float:
    max_error = 0.0
    for start in range(0, hidden.shape[0], 2048):
        stop = min(start + 2048, hidden.shape[0])
        reference = torch.matmul(
            hidden[start:stop].float(), weight.float().transpose(0, 1)
        )
        error = float(
            (actual[start:stop].float() - reference).abs().max().item()
        )
        max_error = max(max_error, error)
    return max_error


def independent_accept_reference(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_predict = torch.argmax(target_logits, dim=-1).view(
        SPATIAL_BATCH, STRIDE
    )
    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    cap = verify_lens.to(torch.int32) - 1
    cap_trim_lens = (correct_len - cap).clamp_min(0)
    correct_len = torch.minimum(correct_len, cap)
    rows = torch.arange(SPATIAL_BATCH, device=target_predict.device)
    bonus = target_predict[rows, correct_len.long()].to(torch.int64)
    return correct_len, bonus, cap_trim_lens.to(torch.int32)


def independent_finalize_reference(
    *,
    correct_len: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    commit_lens = correct_len.to(torch.int32) + 1
    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
    return commit_lens, new_seq_lens, cap_trim_lens.to(torch.int32)


def independent_out_tokens_reference(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
) -> torch.Tensor:
    out_tokens = torch.empty(
        (SPATIAL_BATCH, STRIDE), dtype=torch.int64, device=draft_tokens.device
    )
    out_tokens[:, :GAMMA].copy_(draft_tokens)
    out_tokens[:, GAMMA].fill_(0)
    out_tokens.scatter_(
        1, correct_len.to(torch.int64)[:, None], bonus[:, None]
    )
    return out_tokens


def timed_forward(
    forward: Callable[[], PipelineResult],
) -> tuple[PipelineResult, float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    result = forward()
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    event_seconds = float(start_event.elapsed_time(end_event) / 1000.0)
    return result, event_seconds, wall_seconds


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def git_output(command: list[str]) -> str:
    return subprocess.check_output(command, cwd="/job/sglang", text=True).strip()


def validate_result(
    result: PipelineResult,
    *,
    target_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    draft_hidden: torch.Tensor,
    commit_linears: list[torch.nn.Module],
    candidates: torch.Tensor,
    draft_tokens: torch.Tensor,
    verify_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    baseline: PipelineResult | None,
) -> dict[str, Any]:
    target_projection_error = max_abs_error_fp32(
        result.target_logits, target_hidden, lm_head_weight
    )
    expected_selected = draft_hidden.view(
        SPATIAL_BATCH, STRIDE, DRAFT_HIDDEN_SIZE
    )[:, 1:, :]
    draft_selection_exact = bool(
        torch.equal(
            result.selected_draft_hidden,
            expected_selected.reshape(SPATIAL_BATCH * GAMMA, DRAFT_HIDDEN_SIZE),
        )
        and torch.equal(result.selected_draft_hidden_3d, expected_selected)
    )
    expected_correct, expected_bonus, expected_trim = independent_accept_reference(
        candidates=candidates,
        target_logits=result.target_logits[: SPATIAL_BATCH * STRIDE],
        verify_lens=verify_lens,
    )
    accept_exact = bool(
        torch.equal(result.correct_len, expected_correct)
        and torch.equal(result.bonus, expected_bonus)
        and torch.equal(result.cap_trim_lens, expected_trim)
    )
    expected_commit_lens, expected_new_seq_lens, expected_trim = (
        independent_finalize_reference(
            correct_len=result.correct_len,
            cap_trim_lens=result.cap_trim_lens,
            prefix_lens=prefix_lens,
        )
    )
    finalize_exact = bool(
        torch.equal(result.commit_lens, expected_commit_lens)
        and torch.equal(result.new_seq_lens, expected_new_seq_lens)
        and torch.equal(result.cap_trim_lens, expected_trim)
    )
    expected_out_tokens = independent_out_tokens_reference(
        draft_tokens=draft_tokens,
        correct_len=result.correct_len,
        bonus=result.bonus,
    )
    out_tokens_exact = bool(torch.equal(result.out_tokens, expected_out_tokens))
    commit_errors = [
        max_abs_error_fp32(
            result.committed_kv[index],
            result.selected_draft_hidden,
            commit_linears[index].weight,
        )
        for index in range(COMMIT_STAGES)
    ]
    commit_projection_error = max(commit_errors)

    if baseline is None:
        target_vs_baseline_error = 0.0
        commit_vs_baseline_error = 0.0
        final_state_exact = True
        out_tokens_vs_baseline_exact = True
    else:
        target_vs_baseline_error = float(
            (result.target_logits.float() - baseline.target_logits.float())
            .abs()
            .max()
            .item()
        )
        commit_vs_baseline_error = max(
            float((got.float() - ref.float()).abs().max().item())
            for got, ref in zip(result.committed_kv, baseline.committed_kv)
        )
        final_state_exact = bool(
            torch.equal(result.correct_len, baseline.correct_len)
            and torch.equal(result.bonus, baseline.bonus)
            and torch.equal(result.cap_trim_lens, baseline.cap_trim_lens)
            and torch.equal(result.commit_lens, baseline.commit_lens)
            and torch.equal(result.new_seq_lens, baseline.new_seq_lens)
        )
        out_tokens_vs_baseline_exact = bool(
            torch.equal(result.out_tokens, baseline.out_tokens)
        )

    accepted_tokens = int(result.commit_lens.sum().item())
    useful_tokens = PREFILL_TOKENS + accepted_tokens
    gates = {
        "target_projection_max_abs_le_0.02": (
            target_projection_error <= PROJECTION_MAX_ABS_GATE
        ),
        "commit_projection_max_abs_le_0.02": (
            commit_projection_error <= PROJECTION_MAX_ABS_GATE
        ),
        "draft_hidden_selection_exact": draft_selection_exact,
        "accept_greedy_exact": accept_exact,
        "finalize_accept_exact": finalize_exact,
        "build_out_tokens_exact": out_tokens_exact,
        "target_vs_unchunked_max_abs_le_0.02": (
            target_vs_baseline_error <= PROJECTION_MAX_ABS_GATE
        ),
        "commit_vs_unchunked_max_abs_le_0.02": (
            commit_vs_baseline_error <= PROJECTION_MAX_ABS_GATE
        ),
        "final_state_vs_unchunked_exact": final_state_exact,
        "out_tokens_vs_unchunked_exact": out_tokens_vs_baseline_exact,
        "useful_work_vs_unchunked_exact": (
            baseline is None
            or useful_tokens
            == PREFILL_TOKENS + int(baseline.commit_lens.sum().item())
        ),
    }
    return {
        "accuracy": {
            "target_projection_max_abs_error": target_projection_error,
            "commit_projection_max_abs_error": commit_projection_error,
            "draft_hidden_selection_exact": draft_selection_exact,
            "accept_greedy_exact": accept_exact,
            "finalize_accept_exact": finalize_exact,
            "build_out_tokens_exact": out_tokens_exact,
            "target_vs_unchunked_max_abs_error": target_vs_baseline_error,
            "commit_vs_unchunked_max_abs_error": commit_vs_baseline_error,
            "final_state_vs_unchunked_exact": final_state_exact,
            "out_tokens_vs_unchunked_exact": out_tokens_vs_baseline_exact,
        },
        "work": {
            "target_projected_tokens": PREFILL_TOKENS,
            "commit_projected_rows": SPATIAL_BATCH * GAMMA,
            "accepted_tokens": accepted_tokens,
            "useful_tokens": useful_tokens,
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("results.json")
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("this study requires exactly one CUDA/ROCm GPU")
    if len(CASES) > 6:
        raise RuntimeError("workload case limit is six")

    started_wall = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    target_hidden = (
        torch.randn(
            (PREFILL_TOKENS, TARGET_HIDDEN_SIZE),
            device="cuda",
            generator=generator,
        )
        * 0.25
    ).to(DTYPE)
    lm_head_weight = (
        torch.randn(
            (VOCAB_SIZE, TARGET_HIDDEN_SIZE), device="cuda", generator=generator
        )
        * 0.01
    ).to(DTYPE)
    lm_head = LinearStub(lm_head_weight).cuda()
    draft_hidden = (
        torch.randn(
            (SPATIAL_BATCH * STRIDE, DRAFT_HIDDEN_SIZE),
            device="cuda",
            generator=generator,
        )
        * 0.25
    ).to(DTYPE)
    commit_weights = [
        (
            torch.randn(
                (COMMIT_HEAD_DIM, DRAFT_HIDDEN_SIZE),
                device="cuda",
                generator=generator,
            )
            * 0.02
        ).to(DTYPE)
        for _ in range(COMMIT_STAGES)
    ]
    commit_linears = [LinearStub(weight).cuda() for weight in commit_weights]
    draft_tokens = torch.randint(
        0,
        VOCAB_SIZE,
        (SPATIAL_BATCH, GAMMA),
        device="cuda",
        generator=generator,
    )
    verify_lens = torch.randint(
        1, STRIDE + 1, (SPATIAL_BATCH,), device="cuda", generator=generator
    ).to(torch.int32)
    prefix_lens = torch.randint(
        1, 4096, (SPATIAL_BATCH,), device="cuda", generator=generator
    ).to(torch.int64)

    reference_predictions = torch.empty(
        (PREFILL_TOKENS,), dtype=torch.int64, device="cuda"
    )
    for start in range(0, PREFILL_TOKENS, 2048):
        stop = min(start + 2048, PREFILL_TOKENS)
        reference_logits = torch.matmul(
            target_hidden[start:stop].float(),
            lm_head_weight.float().transpose(0, 1),
        )
        reference_predictions[start:stop] = reference_logits.argmax(dim=-1)
    target_predictions = reference_predictions[
        : SPATIAL_BATCH * STRIDE
    ].view(SPATIAL_BATCH, STRIDE)
    candidates = torch.randint(
        0,
        VOCAB_SIZE,
        (SPATIAL_BATCH, STRIDE),
        device="cuda",
        generator=generator,
    )
    intended_accept_lens = torch.randint(
        0, GAMMA, (SPATIAL_BATCH,), device="cuda", generator=generator
    ).to(torch.int64)
    for row, accept_len in enumerate(intended_accept_lens.tolist()):
        candidates[row, 1 : accept_len + 1] = target_predictions[row, :accept_len]
        if accept_len < GAMMA:
            mismatch = (target_predictions[row, accept_len] + 1) % VOCAB_SIZE
            candidates[row, accept_len + 1] = mismatch
    del reference_predictions, target_predictions, intended_accept_lens

    weight_bytes = module_weight_bytes([lm_head, *commit_linears])
    if weight_bytes >= WEIGHT_LIMIT_BYTES:
        raise RuntimeError(f"generated weights exceed limit: {weight_bytes} bytes")

    case_results: list[dict[str, Any]] = []
    baseline: PipelineResult | None = None
    for case in CASES:
        case_start_allocated = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()

        def forward() -> PipelineResult:
            return run_pipeline(
                target_hidden=target_hidden,
                lm_head=lm_head,
                draft_hidden=draft_hidden,
                commit_linears=commit_linears,
                candidates=candidates,
                draft_tokens=draft_tokens,
                verify_lens=verify_lens,
                prefix_lens=prefix_lens,
                target_chunk=case.target_chunk,
                commit_chunk=case.commit_chunk,
            )

        cold, cold_event_seconds, cold_wall_seconds = timed_forward(forward)
        validation = validate_result(
            cold,
            target_hidden=target_hidden,
            lm_head_weight=lm_head_weight,
            draft_hidden=draft_hidden,
            commit_linears=commit_linears,
            candidates=candidates,
            draft_tokens=draft_tokens,
            verify_lens=verify_lens,
            prefix_lens=prefix_lens,
            baseline=baseline,
        )
        if baseline is None:
            baseline = cold

        warm_event_seconds: list[float] = []
        warm_wall_seconds: list[float] = []
        for _ in range(WARM_FORWARDS):
            _, event_seconds, wall_seconds = timed_forward(forward)
            warm_event_seconds.append(event_seconds)
            warm_wall_seconds.append(wall_seconds)

        median_warm_wall_seconds = statistics.median(warm_wall_seconds)
        useful_tokens = validation["work"]["useful_tokens"]
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
        current_allocated_bytes = int(torch.cuda.memory_allocated())
        gates = dict(validation["gates"])
        gates["weights_under_4gb"] = weight_bytes < WEIGHT_LIMIT_BYTES
        gates["peak_allocations_under_48gb"] = (
            peak_allocated_bytes < PEAK_ALLOCATION_LIMIT_BYTES
        )
        case_results.append(
            {
                "case": {
                    "name": case.name,
                    "target_chunk_rows": case.target_chunk,
                    "target_chunks": PREFILL_TOKENS // case.target_chunk,
                    "commit_chunk_rows": case.commit_chunk,
                    "commit_chunks": SPATIAL_BATCH * GAMMA // case.commit_chunk,
                },
                "accuracy": validation["accuracy"],
                "work": validation["work"],
                "gates": gates,
                "timing": {
                    "cold_cuda_event_seconds": cold_event_seconds,
                    "cold_wall_seconds": cold_wall_seconds,
                    "warm_cuda_event_seconds": warm_event_seconds,
                    "warm_wall_seconds": warm_wall_seconds,
                    "warm_median_wall_seconds": median_warm_wall_seconds,
                    "warm_p99_wall_seconds": percentile(warm_wall_seconds, 0.99),
                    "prefill_tokens_per_second": (
                        PREFILL_TOKENS / median_warm_wall_seconds
                    ),
                    "useful_tokens_per_second": (
                        useful_tokens / median_warm_wall_seconds
                    ),
                    "accepted_tokens_per_second": (
                        validation["work"]["accepted_tokens"]
                        / median_warm_wall_seconds
                    ),
                },
                "resources": {
                    "weight_bytes": weight_bytes,
                    "case_start_allocated_bytes": case_start_allocated,
                    "peak_allocated_bytes": peak_allocated_bytes,
                    "peak_above_case_start_bytes": (
                        peak_allocated_bytes - case_start_allocated
                    ),
                    "current_allocated_bytes": current_allocated_bytes,
                },
            }
        )
        if not all(gates.values()):
            raise RuntimeError(f"case {case.name} failed a gate: {gates}")

    import sglang
    import sgl_kernel
    import triton

    gpu_properties = torch.cuda.get_device_properties(0)
    all_gates_passed = all(
        gate
        for case_result in case_results
        for gate in case_result["gates"].values()
    )
    output = {
        "schema_version": 1,
        "label": (
            "one-GPU gfx942 synthetic DSpark chunk-partition study; "
            "not a GLM, full-model, distributed, or serving claim"
        ),
        "environment": {
            "campaign": "repo-e2e-20260909",
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "gcn_arch": gpu_properties.gcnArchName,
                "count": torch.cuda.device_count(),
                "total_memory_bytes": gpu_properties.total_memory,
            },
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_path": torch.__file__,
            "triton_version": triton.__version__,
            "triton_path": triton.__file__,
            "sglang_version": sglang.__version__,
            "sglang_path": sglang.__file__,
            "sgl_kernel_path": sgl_kernel.__file__,
            "git_commit": git_output(["git", "rev-parse", "HEAD"]),
            "git_branch": git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "image": {
                "requested_tag": (
                    "amdpilotv2/open-job-mi300:"
                    "jit-config-readable-260909-banff5"
                ),
                "operator_provided_local_image_id": (
                    "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1"
                ),
                "inspection_note": (
                    "docker CLI is unavailable in the job container; "
                    "the operator-provided local image ID is recorded verbatim"
                ),
            },
        },
        "workload": {
            "prefill_tokens": PREFILL_TOKENS,
            "spatial_batch": SPATIAL_BATCH,
            "gamma": GAMMA,
            "verify_num_draft_tokens": STRIDE,
            "target_hidden_size": TARGET_HIDDEN_SIZE,
            "draft_hidden_size": DRAFT_HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "commit_stages": COMMIT_STAGES,
            "commit_head_dim": COMMIT_HEAD_DIM,
            "dtype": str(DTYPE),
            "seed": SEED,
            "identical_data_across_cases": True,
            "case_count": len(CASES),
            "warm_forwards_per_case": WARM_FORWARDS,
        },
        "numerical_gates": {
            "target_projection_max_abs": PROJECTION_MAX_ABS_GATE,
            "commit_projection_max_abs": PROJECTION_MAX_ABS_GATE,
            "integer_handoffs_and_final_state": "exact equality",
            "unchanged_from_prior_reduced_pipeline_study": True,
        },
        "commands": {
            "study": (
                "PYTHONPATH=/job/sglang/python /opt/venv/bin/python "
                "reports/j-f037728c0788/chunk_partition.py "
                f"--output {args.output}"
            ),
            "primitive_validation": (
                "PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q "
                "test/registered/spec/dspark/test_dspark_kernel_parity.py "
                "test/registered/spec/utils/test_build_eagle_tree.py"
            ),
        },
        "cases": case_results,
        "totals": {
            "all_gates_passed": all_gates_passed,
            "case_count": len(case_results),
            "cold_forwards": len(case_results),
            "warm_forwards": len(case_results) * WARM_FORWARDS,
            "total_real_forwards": len(case_results) * (1 + WARM_FORWARDS),
            "generated_weight_bytes": weight_bytes,
            "study_wall_seconds_after_import": time.perf_counter() - started_wall,
        },
        "boundaries": {
            "model_specific_constructor": (
                "not instantiated: the GLM/DSpark target-hidden projector "
                "constructor requires a full model configuration and checkpoint"
            ),
            "scope": (
                "reduced target hidden projection, draft hidden selection, "
                "commit-KV projection, greedy acceptance, finalization, and "
                "output-token commit only"
            ),
            "not_included": [
                "GLM weights",
                "attention",
                "KV-cache pool writes",
                "graph replay",
                "scheduler overlap",
                "distributed execution",
                "end-to-end serving",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["totals"], indent=2))
    if not all_gates_passed:
        raise RuntimeError("one or more numerical or resource gates failed")


if __name__ == "__main__":
    main()
