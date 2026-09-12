"""Evidence-based parallel-plan records and startup resolution for diffusion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.perf_logger import get_git_commit_hash

SCHEMA_VERSION = 1
PASSING_QUALITY_GATES = frozenset({"pass", "passed"})


@dataclass(frozen=True)
class DiffusionWorkloadSignature:
    model_revision: str
    task_type: str
    latent_tokens: int
    num_inference_steps: int | tuple[int, ...]
    logical_batch: int
    candidate_count: int
    cfg_branches: int
    dtype: str
    cache_mode: str
    compile_mode: str
    device_topology: str

    def __post_init__(self) -> None:
        if isinstance(self.num_inference_steps, list):
            object.__setattr__(
                self, "num_inference_steps", tuple(self.num_inference_steps)
            )
        for name in (
            "latent_tokens",
            "logical_batch",
            "candidate_count",
            "cfg_branches",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def stable_hash(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class ParallelExecutionPlan:
    num_gpus: int
    cfg_parallel_size: int = 1
    tensor_parallel_size: int = 1
    sequence_parallel_size: int = 1
    fsdp: bool = False
    data_parallel_replicas: int = 1

    def __post_init__(self) -> None:
        for name in (
            "num_gpus",
            "cfg_parallel_size",
            "tensor_parallel_size",
            "sequence_parallel_size",
            "data_parallel_replicas",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def required_devices(self) -> int:
        return (
            self.cfg_parallel_size
            * self.tensor_parallel_size
            * self.sequence_parallel_size
            * self.data_parallel_replicas
        )


@dataclass(frozen=True)
class CalibrationResult:
    signature: DiffusionWorkloadSignature
    plan: ParallelExecutionPlan
    feasible: bool
    latency_ms: Mapping[str, float]
    peak_memory_bytes: int
    quality_gate: str
    communication_summary: Mapping[str, float]
    throughput: float | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class CalibrationReport:
    code_revision: str
    environment: Mapping[str, Any]
    results: tuple[CalibrationResult, ...]
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def load(cls, path: str | Path) -> CalibrationReport:
        raw = json.loads(Path(path).read_text())
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported calibration schema {raw.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        results = []
        for item in raw.get("results", ()):
            item = dict(item)
            recorded_hash = item.pop("signature_hash", None)
            signature = DiffusionWorkloadSignature(**item["signature"])
            if recorded_hash is not None and recorded_hash != signature.stable_hash:
                raise ValueError(
                    "calibration result signature_hash does not match payload"
                )
            results.append(
                CalibrationResult(
                    **{
                        **item,
                        "signature": signature,
                        "plan": ParallelExecutionPlan(**item["plan"]),
                    }
                )
            )
        return cls(
            schema_version=raw["schema_version"],
            code_revision=raw["code_revision"],
            environment=raw.get("environment", {}),
            results=tuple(results),
        )

    def dump(self, path: str | Path) -> None:
        payload = asdict(self)
        for encoded, result in zip(payload["results"], self.results):
            encoded["signature_hash"] = result.signature.stable_hash
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class PlanCapabilities:
    num_devices: int
    cfg_parallel_sizes: tuple[int, ...] = (1,)
    tensor_parallel_sizes: tuple[int, ...] = (1,)
    sequence_parallel_sizes: tuple[int, ...] = (1,)
    data_parallel_replicas: tuple[int, ...] = (1,)
    supports_fsdp: bool = False
    attention_heads: int | None = None
    available_memory_bytes: int | None = None


def validate_plan(
    plan: ParallelExecutionPlan,
    signature: DiffusionWorkloadSignature,
    capabilities: PlanCapabilities,
    estimated_peak_memory_bytes: int | None = None,
) -> tuple[str, ...]:
    reasons = []
    if plan.num_gpus != capabilities.num_devices:
        reasons.append("plan num_gpus does not match selected devices")
    if plan.required_devices != plan.num_gpus:
        reasons.append("parallel degree product does not equal num_gpus")
    for value, allowed, name in (
        (plan.cfg_parallel_size, capabilities.cfg_parallel_sizes, "CFG"),
        (plan.tensor_parallel_size, capabilities.tensor_parallel_sizes, "TP"),
        (plan.sequence_parallel_size, capabilities.sequence_parallel_sizes, "SP"),
        (plan.data_parallel_replicas, capabilities.data_parallel_replicas, "DP"),
    ):
        if value not in allowed:
            reasons.append(f"unsupported {name} degree {value}")
    if plan.cfg_parallel_size > signature.cfg_branches:
        reasons.append("CFG degree exceeds workload branches")
    if capabilities.attention_heads is not None and (
        capabilities.attention_heads % plan.tensor_parallel_size
    ):
        reasons.append("attention heads are not divisible by TP degree")
    if signature.latent_tokens % plan.sequence_parallel_size:
        reasons.append("latent tokens are not divisible by SP degree")
    if plan.fsdp and not capabilities.supports_fsdp:
        reasons.append("FSDP is unsupported")
    if (
        estimated_peak_memory_bytes is not None
        and capabilities.available_memory_bytes is not None
        and estimated_peak_memory_bytes > capabilities.available_memory_bytes
    ):
        reasons.append("estimated peak memory exceeds least-free device")
    return tuple(reasons)


def enumerate_legal_plans(
    signature: DiffusionWorkloadSignature, capabilities: PlanCapabilities
) -> tuple[ParallelExecutionPlan, ...]:
    plans = []
    for cfg in capabilities.cfg_parallel_sizes:
        for tp in capabilities.tensor_parallel_sizes:
            for sp in capabilities.sequence_parallel_sizes:
                for dp in capabilities.data_parallel_replicas:
                    fsdp_options = (
                        (False, True) if capabilities.supports_fsdp else (False,)
                    )
                    for fsdp in fsdp_options:
                        plan = ParallelExecutionPlan(
                            capabilities.num_devices, cfg, tp, sp, fsdp, dp
                        )
                        if not validate_plan(plan, signature, capabilities):
                            plans.append(plan)
    return tuple(plans)


@dataclass(frozen=True)
class Resolution:
    plan: ParallelExecutionPlan | None
    source: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    signature_hash: str | None = None


def get_runtime_environment() -> dict[str, str]:
    """Return the runtime identity fields required for calibration reuse."""
    runtime_version = (
        torch.version.hip
        or torch.version.cuda
        or getattr(torch.version, "xpu", None)
        or "none"
    )
    try:
        device_name = str(current_platform.get_device_name(0))
    except Exception:
        device_name = str(current_platform.device_type)
    try:
        distributed_backend = current_platform.get_torch_distributed_backend_str()
    except Exception:
        distributed_backend = "unknown"
    return {
        "device_type": str(current_platform.device_type),
        "device_name": device_name,
        "distributed_backend": str(distributed_backend),
        "torch_version": str(torch.__version__),
        "accelerator_runtime_version": str(runtime_version),
    }


def _intrinsic_plan_rejection(plan: ParallelExecutionPlan) -> str | None:
    if plan.required_devices != plan.num_gpus:
        return "parallel degree product does not equal num_gpus"
    return None


def resolve_calibrated_plan(
    report: CalibrationReport,
    signature: DiffusionWorkloadSignature,
    *,
    objective: str = "p95",
    code_revision: str | None = None,
    environment: Mapping[str, Any] | None = None,
) -> Resolution:
    if code_revision is not None and report.code_revision != code_revision:
        return Resolution(None, "conservative_fallback", ("code revision mismatch",))
    if environment is not None and any(
        report.environment.get(key) != value for key, value in environment.items()
    ):
        return Resolution(
            None,
            "conservative_fallback",
            ("calibration environment mismatch",),
        )
    exact = [result for result in report.results if result.signature == signature]
    if not exact:
        return Resolution(None, "conservative_fallback", ("no exact signature match",))
    passing = []
    rejected = []
    for result in exact:
        if not result.feasible:
            rejected.append(
                result.rejection_reason or "static/runtime feasibility failed"
            )
        elif result.quality_gate.lower() not in PASSING_QUALITY_GATES:
            rejected.append(f"quality gate={result.quality_gate}")
        elif objective not in result.latency_ms:
            rejected.append(f"missing latency objective {objective}")
        elif reason := _intrinsic_plan_rejection(result.plan):
            rejected.append(reason)
        else:
            passing.append(result)
    if not passing:
        return Resolution(None, "conservative_fallback", tuple(rejected))
    winner = min(passing, key=lambda result: result.latency_ms[objective])
    return Resolution(winner.plan, "exact_calibration", tuple(rejected))


PARALLEL_EXPLICIT_ARGS = frozenset(
    {
        "num_gpus",
        "tp_size",
        "sp_degree",
        "cfg_parallel_degree",
        "enable_cfg_parallel",
        "dp_size",
        "use_fsdp_inference",
    }
)


def apply_advisor_to_server_args(server_args: Any) -> Resolution:
    spec = server_args.diffusion_parallel_plan
    if not spec:
        return Resolution(None, "disabled")
    explicit = PARALLEL_EXPLICIT_ARGS.intersection(server_args._explicit_arg_names)
    if explicit:
        reason = f"explicit parallel flags: {', '.join(sorted(explicit))}"
        return Resolution(None, "explicit_flags", (reason,))
    if not spec.startswith("auto:"):
        raise ValueError("--diffusion-parallel-plan must be auto:<report.json>")
    if not server_args.diffusion_workload_signature:
        raise ValueError("--diffusion-workload-signature is required with the advisor")
    signature_arg = server_args.diffusion_workload_signature
    signature_raw = (
        json.loads(signature_arg)
        if signature_arg.lstrip().startswith("{")
        else json.loads(Path(signature_arg).read_text())
    )
    if not isinstance(signature_raw, dict):
        raise ValueError("workload signature JSON must be an object")
    signature = DiffusionWorkloadSignature(**signature_raw)
    resolution = resolve_calibrated_plan(
        CalibrationReport.load(spec[5:]),
        signature,
        code_revision=get_git_commit_hash(),
        environment=get_runtime_environment(),
    )
    resolution = Resolution(
        resolution.plan,
        resolution.source,
        resolution.reasons,
        signature.stable_hash,
    )
    if resolution.plan is None:
        return resolution
    plan = resolution.plan
    server_args.num_gpus = plan.num_gpus
    server_args.tp_size = plan.tensor_parallel_size
    server_args.sp_degree = plan.sequence_parallel_size
    server_args.cfg_parallel_degree = plan.cfg_parallel_size
    server_args.enable_cfg_parallel = plan.cfg_parallel_size > 1
    server_args.dp_size = plan.data_parallel_replicas
    server_args.use_fsdp_inference = plan.fsdp
    return resolution
