# SPDX-License-Identifier: Apache-2.0
"""Offline trajectory gates and runtime resolution for promoted compile plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CompileWorkloadSignature:
    model_revision: str
    dtype: str
    backend: str
    parallel_signature: str
    latent_shape_regime: tuple[int, ...]
    num_inference_steps: int | tuple[int, ...]
    cfg_mode: str
    cache_mode: str
    state_schema_version: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class TrajectoryGate:
    checkpoints: tuple[str, ...]
    tensor_thresholds: Mapping[str, Mapping[str, float]]
    output_metrics: Mapping[str, float] = field(default_factory=dict)
    require_decision_trace_match: bool = True

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class CompiledPlanManifest:
    signature: CompileWorkloadSignature
    regions: tuple[str, ...]
    compile_options: Mapping[str, object]
    gate_digest: str
    status: Literal["validated", "rejected"]
    region_digest: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompiledPlanManifest":
        if value.get("schema_version", 1) != 1:
            raise ValueError(f"Unsupported compile manifest schema: {value.get('schema_version')}")
        signature = dict(value["signature"])
        signature["latent_shape_regime"] = tuple(signature["latent_shape_regime"])
        steps = signature["num_inference_steps"]
        if isinstance(steps, list):
            signature["num_inference_steps"] = tuple(steps)
        return cls(
            signature=CompileWorkloadSignature(**signature),
            regions=tuple(value["regions"]),
            compile_options=dict(value["compile_options"]),
            gate_digest=str(value["gate_digest"]),
            status=value["status"],
            region_digest=str(value["region_digest"]),
            schema_version=1,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompiledPlanManifest":
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def write(self, path: str | Path) -> None:
        Path(path).write_text(_canonical_json(self.to_dict()) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CompilePlanResolution:
    manifest: CompiledPlanManifest | None
    fallback_reason: str | None

    @property
    def use_compiled(self) -> bool:
        return self.manifest is not None


class CompilePlanResolver:
    def __init__(self, manifests: Sequence[CompiledPlanManifest]) -> None:
        self._manifests = tuple(manifests)

    @classmethod
    def load(cls, paths: Sequence[str | Path]) -> "CompilePlanResolver":
        return cls([CompiledPlanManifest.load(path) for path in paths])

    def resolve(
        self,
        signature: CompileWorkloadSignature,
        *,
        region_digest: str,
        gate_digest: str,
    ) -> CompilePlanResolution:
        candidates = [m for m in self._manifests if m.signature == signature]
        if not candidates:
            return CompilePlanResolution(None, "workload_signature_mismatch")
        manifest = candidates[0]
        if manifest.status != "validated":
            return CompilePlanResolution(None, "plan_rejected")
        if manifest.region_digest != region_digest:
            return CompilePlanResolution(None, "region_digest_mismatch")
        if manifest.gate_digest != gate_digest:
            return CompilePlanResolution(None, "gate_digest_mismatch")
        return CompilePlanResolution(manifest, None)


@dataclass
class TrajectoryCapture:
    checkpoints: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    decision_trace: list[str] = field(default_factory=list)
    terminal_state: Mapping[str, object] = field(default_factory=dict)
    outputs: Mapping[str, object] = field(default_factory=dict)

    def record(self, name: str, tensor: torch.Tensor) -> None:
        # Cloning prevents a later step/request from mutating captured evidence.
        self.checkpoints.setdefault(name, []).append(tensor.detach().cpu().clone())


@dataclass(frozen=True)
class TrajectoryGateResult:
    passed: bool
    failures: tuple[str, ...]
    checkpoint_metrics: Mapping[str, tuple[Mapping[str, float], ...]]
    output_metrics: Mapping[str, float]


def _tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    lhs = reference.detach().cpu().double()
    rhs = candidate.detach().cpu().double()
    if lhs.shape != rhs.shape:
        raise ValueError(f"shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}")
    diff = (lhs - rhs).abs()
    denom = lhs.abs().clamp_min(torch.finfo(torch.float64).eps)
    flat_lhs, flat_rhs = lhs.flatten(), rhs.flatten()
    cosine = torch.nn.functional.cosine_similarity(flat_lhs, flat_rhs, dim=0)
    return {
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mae": float(diff.mean()) if diff.numel() else 0.0,
        "max_relative": float((diff / denom).max()) if diff.numel() else 0.0,
        "cosine_similarity": float(cosine) if diff.numel() else 1.0,
    }


def evaluate_trajectory_gate(
    reference: TrajectoryCapture,
    candidate: TrajectoryCapture,
    gate: TrajectoryGate,
    *,
    output_metric_adapters: Mapping[str, Callable[[object, object], float]] | None = None,
) -> TrajectoryGateResult:
    failures: list[str] = []
    reports: dict[str, tuple[Mapping[str, float], ...]] = {}
    for name in gate.checkpoints:
        ref_steps = reference.checkpoints.get(name)
        cand_steps = candidate.checkpoints.get(name)
        if ref_steps is None or cand_steps is None:
            failures.append(f"missing_checkpoint:{name}")
            continue
        if len(ref_steps) != len(cand_steps):
            failures.append(f"step_count:{name}")
            continue
        step_reports: list[Mapping[str, float]] = []
        for step, (lhs, rhs) in enumerate(zip(ref_steps, cand_steps, strict=True)):
            try:
                metrics = _tensor_metrics(lhs, rhs)
            except ValueError:
                failures.append(f"shape:{name}:{step}")
                continue
            step_reports.append(metrics)
            for metric, threshold in gate.tensor_thresholds.get(name, {}).items():
                value = metrics[metric]
                failed = value < threshold if metric == "cosine_similarity" else value > threshold
                if failed:
                    failures.append(f"tensor:{name}:{step}:{metric}")
        reports[name] = tuple(step_reports)
    if gate.require_decision_trace_match and reference.decision_trace != candidate.decision_trace:
        failures.append("decision_trace")
    if reference.terminal_state != candidate.terminal_state:
        failures.append("terminal_state")
    adapters = output_metric_adapters or {}
    output_report: dict[str, float] = {}
    for name, threshold in gate.output_metrics.items():
        if name not in adapters or name not in reference.outputs or name not in candidate.outputs:
            failures.append(f"missing_output_metric:{name}")
            continue
        value = adapters[name](reference.outputs[name], candidate.outputs[name])
        output_report[name] = value
        if value < threshold:
            failures.append(f"output_metric:{name}")
    return TrajectoryGateResult(not failures, tuple(failures), reports, output_report)
