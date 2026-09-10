from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    FinalizeAcceptLens,
)
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import CommitKvProj
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildCommitInjectLayout,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    select_draft_hidden_without_anchor,
)


SEED = 30734
WARMUP_FORWARDS = 3
MEASURED_FORWARDS = 10
TIMING_REPEATS = 5
WEIGHT_LIMIT_BYTES = 4 * 1024**3
LIVE_ALLOCATION_LIMIT_BYTES = 48 * 1024**3
WALL_LIMIT_SECONDS = 7200


@dataclass(frozen=True)
class WorkloadCase:
    name: str
    batch_size: int
    gamma: int
    hidden_size: int
    head_dim: int
    stages: int
    request_pool_size: int
    request_pool_length: int


WORKLOAD_CASES = [
    WorkloadCase("small_block", 8, 4, 1024, 128, 2, 32, 512),
    WorkloadCase("medium_block", 32, 4, 2048, 576, 3, 64, 1024),
    WorkloadCase("wide_batch", 64, 6, 1024, 128, 2, 128, 2048),
    WorkloadCase("large_token_block", 128, 4, 512, 128, 2, 256, 4096),
]


CONFIGURATIONS = [
    "torch_layout_torch_projection",
    "triton_layout_torch_projection",
    "torch_layout_triton_projection",
    "triton_layout_triton_projection",
]


class SyntheticProjection(torch.nn.Module):
    quant_method = None

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(hidden, self.weight), None


def _layout_method(configuration: str):
    return (
        BuildCommitInjectLayout.triton
        if configuration.startswith("triton_layout")
        else BuildCommitInjectLayout.torch
    )


def _projection_method(configuration: str):
    return (
        CommitKvProj.triton
        if configuration.endswith("triton_projection")
        else CommitKvProj.torch
    )


def _commit_block(
    *,
    configuration: str,
    target_hidden: torch.Tensor,
    projection_linears: list[SyntheticProjection],
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> tuple[Any, list[torch.Tensor]]:
    layout = _layout_method(configuration)(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        prefix_lens=prefix_lens,
        block_pos_offsets=block_pos_offsets,
        full_to_swa_mapping=full_to_swa_mapping,
        commit_lens=commit_lens,
        stride=stride,
    )
    projected_kv = _projection_method(configuration)(
        main_x=target_hidden,
        wkv_linears=projection_linears,
    )
    return layout, projected_kv


def _reference_accept(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_lens: torch.Tensor,
    batch_size: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_predict = torch.argmax(target_logits, dim=-1).view(batch_size, stride)
    matches = candidates[:, 1:] == target_predict[:, :-1]
    raw_correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    capped_len = torch.minimum(raw_correct_len, verify_lens - 1)
    cap_trim_lens = raw_correct_len - capped_len
    row_ids = torch.arange(batch_size, device=target_predict.device)
    bonus = target_predict[row_ids, capped_len.long()].to(torch.int64)
    return capped_len.to(torch.int32), bonus, cap_trim_lens.to(torch.int32)


def _reference_finalize(
    *,
    correct_len: torch.Tensor,
    cap_trim_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    commit_lens = correct_len.to(torch.int32) + 1
    new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
    return commit_lens, new_seq_lens, cap_trim_lens.to(torch.int32)


def _reference_layout(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets[:stride]
    positions = positions_2d.reshape(-1).to(torch.int64)
    full_locations = req_to_token[
        req_pool_indices.view(-1, 1), positions.view(-1, stride)
    ]
    swa_locations = full_to_swa_mapping[full_locations].to(torch.int32)
    columns = torch.arange(stride, device=swa_locations.device).view(1, -1)
    committed = columns < commit_lens.to(torch.int64).view(-1, 1)
    swa_locations = torch.where(
        committed.reshape(-1), swa_locations.reshape(-1), torch.full_like(swa_locations.reshape(-1), -1)
    )
    return swa_locations, positions


def _reference_projection(
    *,
    target_hidden: torch.Tensor,
    projection_linears: list[SyntheticProjection],
) -> list[torch.Tensor]:
    hidden_float = target_hidden.float()
    return [
        torch.nn.functional.linear(hidden_float, projection.weight.float())
        for projection in projection_linears
    ]


def _exact_gate(name: str, got: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    passed = got.dtype == reference.dtype and got.shape == reference.shape and torch.equal(got, reference)
    return {
        "gate": name,
        "passed": bool(passed),
        "comparison": "exact dtype, shape, and value equality",
    }


def _tolerance_gate(
    *,
    name: str,
    got: list[torch.Tensor],
    reference: list[torch.Tensor],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    passed = len(got) == len(reference)
    if passed:
        for produced, expected in zip(got, reference):
            passed = (
                passed
                and produced.shape == expected.shape
                and bool(torch.isfinite(produced).all())
                and torch.allclose(produced.float(), expected, rtol=rtol, atol=atol)
            )
    return {
        "gate": name,
        "passed": bool(passed),
        "comparison": f"finite allclose rtol={rtol} atol={atol}",
    }


def _time_configuration(
    *,
    configuration: str,
    state: dict[str, Any],
) -> float:
    for _ in range(WARMUP_FORWARDS):
        _commit_block(configuration=configuration, **state["commit_inputs"])
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(MEASURED_FORWARDS):
        _commit_block(configuration=configuration, **state["commit_inputs"])
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / MEASURED_FORWARDS


def _build_case(
    workload: WorkloadCase, case_seed: int
) -> tuple[dict[str, Any], list[SyntheticProjection], int]:
    generator = torch.Generator(device="cuda").manual_seed(case_seed)
    stride = workload.gamma + 1
    token_count = workload.batch_size * stride
    target_hidden = (
        torch.randn(
            (token_count, workload.hidden_size),
            device="cuda",
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16)
    candidates = torch.randint(
        0,
        256,
        (workload.batch_size, stride),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )
    target_logits = torch.randn(
        (token_count, 256), device="cuda", generator=generator
    )
    verify_lens = torch.randint(
        1,
        stride + 1,
        (workload.batch_size,),
        device="cuda",
        dtype=torch.int32,
        generator=generator,
    )
    prefix_lens = torch.randint(
        1,
        workload.request_pool_length - stride,
        (workload.batch_size,),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )
    req_pool_indices = torch.randperm(
        workload.request_pool_size, device="cuda", generator=generator
    )[: workload.batch_size].to(torch.int64)
    req_to_token = torch.randint(
        0,
        1_000_000,
        (workload.request_pool_size, workload.request_pool_length),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )
    full_to_swa_mapping = torch.randint(
        0,
        1_000_000,
        (1_000_000,),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )
    block_pos_offsets = torch.arange(stride, device="cuda", dtype=torch.int64)
    projection_linears = [
        SyntheticProjection(
            (
                torch.randn(
                    (workload.head_dim, workload.hidden_size),
                    device="cuda",
                    generator=generator,
                )
                * 0.02
            ).to(torch.bfloat16)
        )
        for _ in range(workload.stages)
    ]
    weight_bytes = sum(
        projection.weight.numel() * projection.weight.element_size()
        for projection in projection_linears
    )
    state = {
        "target_hidden": target_hidden,
        "candidates": candidates,
        "target_logits": target_logits,
        "verify_lens": verify_lens,
        "prefix_lens": prefix_lens,
        "req_pool_indices": req_pool_indices,
        "req_to_token": req_to_token,
        "full_to_swa_mapping": full_to_swa_mapping,
        "block_pos_offsets": block_pos_offsets,
        "projection_linears": projection_linears,
        "commit_inputs": {
            "target_hidden": target_hidden,
            "projection_linears": projection_linears,
            "req_pool_indices": req_pool_indices,
            "req_to_token": req_to_token,
            "prefix_lens": prefix_lens,
            "block_pos_offsets": block_pos_offsets,
            "full_to_swa_mapping": full_to_swa_mapping,
            "commit_lens": None,
            "stride": stride,
        },
    }
    return state, projection_linears, weight_bytes


def _run_case(
    *,
    workload: WorkloadCase,
    case_seed: int,
    started_at: float,
) -> dict[str, Any]:
    if time.monotonic() - started_at > WALL_LIMIT_SECONDS:
        raise RuntimeError("7200-second wall limit exceeded")
    torch.cuda.reset_peak_memory_stats()
    state, projection_linears, weight_bytes = _build_case(workload, case_seed)
    if weight_bytes >= WEIGHT_LIMIT_BYTES:
        raise RuntimeError(f"synthetic weights exceed 4GB: {weight_bytes}")

    model_hidden, draft_hidden_3d = select_draft_hidden_without_anchor(
        state["target_hidden"],
        bs=workload.batch_size,
        gamma=workload.gamma,
    )
    hidden_reference = (
        state["target_hidden"]
        .view(workload.batch_size, workload.gamma + 1, workload.hidden_size)[:, 1:]
        .contiguous()
    )
    hidden_gate = _exact_gate("draft_hidden_selection", draft_hidden_3d, hidden_reference)

    correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
        candidates=state["candidates"],
        target_logits=state["target_logits"],
        verify_num_draft_tokens=workload.gamma + 1,
        cutoff_verify_lens=state["verify_lens"],
    )
    reference_correct_len, reference_bonus, reference_cap_trim = _reference_accept(
        candidates=state["candidates"],
        target_logits=state["target_logits"],
        verify_lens=state["verify_lens"],
        batch_size=workload.batch_size,
        stride=workload.gamma + 1,
    )
    accept_gates = [
        _exact_gate("accept_correct_len", correct_len, reference_correct_len),
        _exact_gate("accept_bonus", bonus, reference_bonus),
        _exact_gate("accept_cap_trim", cap_trim_lens, reference_cap_trim),
    ]

    finalized = FinalizeAcceptLens.execute(
        correct_len=correct_len,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=state["prefix_lens"],
    )
    reference_commit_lens, reference_new_seq_lens, reference_cap_trim = _reference_finalize(
        correct_len=reference_correct_len,
        cap_trim_lens=reference_cap_trim,
        prefix_lens=state["prefix_lens"],
    )
    finalize_gates = [
        _exact_gate("commit_lens", finalized.commit_lens, reference_commit_lens),
        _exact_gate("new_seq_lens", finalized.new_seq_lens, reference_new_seq_lens),
        _exact_gate("finalize_cap_trim", finalized.cap_trim_lens, reference_cap_trim),
    ]
    state["commit_inputs"]["commit_lens"] = finalized.commit_lens

    reference_swa_locations, reference_positions = _reference_layout(
        req_pool_indices=state["req_pool_indices"],
        req_to_token=state["req_to_token"],
        prefix_lens=state["prefix_lens"],
        block_pos_offsets=state["block_pos_offsets"],
        full_to_swa_mapping=state["full_to_swa_mapping"],
        commit_lens=reference_commit_lens,
        stride=workload.gamma + 1,
    )
    reference_projection = _reference_projection(
        target_hidden=state["target_hidden"],
        projection_linears=projection_linears,
    )

    configuration_results = []
    for configuration in CONFIGURATIONS:
        layout, projected_kv = _commit_block(
            configuration=configuration, **state["commit_inputs"]
        )
        gates = [
            hidden_gate,
            *accept_gates,
            *finalize_gates,
            _exact_gate("commit_swa_loc", layout.swa_loc, reference_swa_locations),
            _exact_gate("commit_positions", layout.positions, reference_positions),
            _tolerance_gate(
                name="commit_kv_projection",
                got=projected_kv,
                reference=reference_projection,
                rtol=2e-2,
                atol=2e-3,
            ),
        ]
        gate_passed = all(gate["passed"] for gate in gates)
        samples_ms = []
        if gate_passed:
            for _ in range(TIMING_REPEATS):
                samples_ms.append(
                    _time_configuration(configuration=configuration, state=state)
                )
        configuration_results.append(
            {
                "configuration": configuration,
                "numerical_gate_passed": gate_passed,
                "numerical_regression_rejected": not gate_passed,
                "gates": gates,
                "timing_samples_ms": samples_ms,
                "timing_median_ms": (
                    statistics.median(samples_ms) if samples_ms else None
                ),
                "timing_iqr_ms": (
                    statistics.quantiles(samples_ms, n=4)[2]
                    - statistics.quantiles(samples_ms, n=4)[0]
                    if len(samples_ms) >= 2
                    else None
                ),
                "timing_min_ms": min(samples_ms) if samples_ms else None,
                "timing_max_ms": max(samples_ms) if samples_ms else None,
            }
        )

    peak_memory_bytes = torch.cuda.max_memory_allocated()
    if peak_memory_bytes >= LIVE_ALLOCATION_LIMIT_BYTES:
        raise RuntimeError(f"live allocations exceed 48GB: {peak_memory_bytes}")
    return {
        "case": workload.name,
        "dimensions": {
            "batch_size": workload.batch_size,
            "gamma": workload.gamma,
            "verify_stride": workload.gamma + 1,
            "target_hidden_size": workload.hidden_size,
            "kv_head_dim": workload.head_dim,
            "draft_stages": workload.stages,
            "request_pool_size": workload.request_pool_size,
            "request_pool_length": workload.request_pool_length,
            "target_hidden_rows": workload.batch_size * (workload.gamma + 1),
            "draft_hidden_rows": workload.batch_size * workload.gamma,
        },
        "dtypes": {
            "target_hidden": "torch.bfloat16",
            "target_logits": "torch.float32",
            "candidates": "torch.int64",
            "verify_lens": "torch.int32",
            "prefix_lens": "torch.int64",
            "req_to_token": "torch.int64",
            "full_to_swa_mapping": "torch.int64",
            "projection_weights": "torch.bfloat16",
        },
        "seed": case_seed,
        "synthetic_weight_bytes": weight_bytes,
        "peak_memory_allocated_bytes": peak_memory_bytes,
        "retained_linears": projection_linears,
        "configurations": configuration_results,
    }


def _collect_environment() -> dict[str, Any]:
    import sglang
    import sgl_kernel
    import triton

    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
    native_paths = [
        "/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so",
        "/opt/rocm/lib/libamdhip64.so.7",
        "/opt/rocm/lib/librccl.so.1",
    ]
    missing_native_paths = [path for path in native_paths if not Path(path).exists()]
    return {
        "source_commit": commit,
        "python_path": "/opt/venv/bin/python",
        "source_paths": {
            "sglang": sglang.__file__,
            "sgl_kernel": sgl_kernel.__file__,
            "triton": triton.__file__,
            "torch": torch.__file__,
        },
        "native_paths": native_paths,
        "missing_native_paths": missing_native_paths,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_count": torch.cuda.device_count(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/j-66135d2994b3/results.json",
        type=Path,
    )
    args = parser.parse_args()
    started_at = time.monotonic()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one GPU, found {torch.cuda.device_count()}")
    torch.manual_seed(SEED)
    retained_linears: list[SyntheticProjection] = []
    case_results = []
    for case_index, workload in enumerate(WORKLOAD_CASES):
        result = _run_case(
            workload=workload,
            case_seed=SEED + case_index,
            started_at=started_at,
        )
        retained_linears.extend(result.pop("retained_linears", []))
        case_results.append(result)

    all_gates_passed = all(
        configuration["numerical_gate_passed"]
        for case_result in case_results
        for configuration in case_result["configurations"]
    )
    output = {
        "study": "reduced DSpark draft/target hidden-state projection and commit block",
        "boundary_label": (
            "model-specific constructor boundary: no GLM/DSpark model constructor, "
            "checkpoint, or distributed path is used; synthetic states exercise the "
            "existing primitive handoffs only"
        ),
        "weights_source": "locally generated synthetic weights and states",
        "distributed_claim": False,
        "world_size": 1,
        "environment": _collect_environment(),
        "image": {
            "qualified_image": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
            "local_image_id_sha256": "39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        },
        "measured_block": "commit inject layout plus stacked/per-linear KV projection",
        "configuration_count": len(CONFIGURATIONS),
        "configurations": CONFIGURATIONS,
        "workload_case_count": len(WORKLOAD_CASES),
        "timing_method": {
            "event": "CUDA events",
            "warmup_forwards_per_sample": WARMUP_FORWARDS,
            "measured_forwards_per_sample": MEASURED_FORWARDS,
            "interleaved_repeats": TIMING_REPEATS,
            "uncertainty": (
                "median, IQR when available, min, and max of interleaved sample means "
                "on one shared MI300X; no dedicated-hardware or distribution-free claim"
            ),
        },
        "accuracy_gates": {
            "hidden_selection": "exact",
            "greedy_accept": "exact",
            "finalize_commit_lens": "exact",
            "commit_inject_layout": "exact",
            "kv_projection": "finite allclose rtol=2e-2 atol=2e-3",
            "policy": "any failed gate rejects the configuration and prevents timing acceptance",
        },
        "limits": {
            "wall_seconds": WALL_LIMIT_SECONDS,
            "synthetic_weight_limit_bytes": WEIGHT_LIMIT_BYTES,
            "live_allocation_limit_bytes": LIVE_ALLOCATION_LIMIT_BYTES,
            "maximum_workload_cases": 6,
            "maximum_configurations": 4,
        },
        "all_numerical_gates_passed": all_gates_passed,
        "final_memory_allocated_bytes": torch.cuda.memory_allocated(),
        "elapsed_seconds": time.monotonic() - started_at,
        "results": case_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "all_numerical_gates_passed": all_gates_passed,
        "elapsed_seconds": output["elapsed_seconds"],
        "final_memory_allocated_bytes": output["final_memory_allocated_bytes"],
    }, indent=2))
    return 0 if all_gates_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
