import json

import pytest
import torch

from sglang.multimodal_gen.runtime.utils.compile_trajectory import (
    CompiledPlanManifest,
    CompilePlanResolver,
    CompileWorkloadSignature,
    TrajectoryCapture,
    TrajectoryGate,
    evaluate_trajectory_gate,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import region_inventory_digest


def _signature(**changes):
    values = dict(
        model_revision="model@abc",
        dtype="bfloat16",
        backend="inductor",
        parallel_signature="tp=1,sp=1,cfg=1",
        latent_shape_regime=(1, 16, 8, 8),
        num_inference_steps=3,
        cfg_mode="serial",
        cache_mode="off",
        state_schema_version="v1",
    )
    values.update(changes)
    return CompileWorkloadSignature(**values)


def _manifest(signature=None, status="validated"):
    regions = ("blocks.0", "blocks.1")
    gate = TrajectoryGate(("latents",), {"latents": {"max_abs": 1e-4}})
    return CompiledPlanManifest(
        signature=signature or _signature(),
        regions=regions,
        compile_options={"backend": "inductor", "fullgraph": False},
        gate_digest=gate.digest,
        status=status,
        region_digest=region_inventory_digest(regions),
    ), gate


def test_signature_hash_and_manifest_json_are_stable(tmp_path):
    signature = _signature()
    reordered = _signature()
    assert signature.digest == reordered.digest
    manifest, _ = _manifest(signature)
    path = tmp_path / "plan.json"
    manifest.write(path)
    assert CompiledPlanManifest.load(path) == manifest
    assert json.loads(path.read_text())["signature"]["latent_shape_regime"] == [
        1,
        16,
        8,
        8,
    ]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"latent_shape_regime": (1, 16, 16, 16)}, "workload_signature_mismatch"),
        ({"model_revision": "model@def"}, "workload_signature_mismatch"),
        ({"state_schema_version": "v2"}, "workload_signature_mismatch"),
    ],
)
def test_resolver_falls_back_outside_validated_regime(change, reason):
    manifest, gate = _manifest()
    result = CompilePlanResolver([manifest]).resolve(
        _signature(**change),
        region_digest=manifest.region_digest,
        gate_digest=gate.digest,
    )
    assert not result.use_compiled
    assert result.fallback_reason == reason


def test_resolver_rejects_stale_region_and_gate_digests():
    manifest, gate = _manifest()
    resolver = CompilePlanResolver([manifest])
    assert (
        resolver.resolve(
            _signature(), region_digest="stale", gate_digest=gate.digest
        ).fallback_reason
        == "region_digest_mismatch"
    )
    assert (
        resolver.resolve(
            _signature(), region_digest=manifest.region_digest, gate_digest="stale"
        ).fallback_reason
        == "gate_digest_mismatch"
    )


def test_resolver_never_promotes_rejected_manifest():
    manifest, gate = _manifest(status="rejected")
    result = CompilePlanResolver([manifest]).resolve(
        _signature(),
        region_digest=manifest.region_digest,
        gate_digest=gate.digest,
    )
    assert result.fallback_reason == "plan_rejected"


def test_capture_clones_request_owned_state():
    capture = TrajectoryCapture()
    state = torch.tensor([1.0])
    capture.record("cache", state)
    state.add_(10)
    assert capture.checkpoints["cache"][0].item() == 1.0


def test_gate_reports_first_perturbed_checkpoint_and_structural_drift():
    reference, candidate = TrajectoryCapture(), TrajectoryCapture()
    for value in (0.0, 1.0, 2.0):
        reference.record("latents", torch.tensor([value, value + 1]))
        candidate.record("latents", torch.tensor([value, value + 1]))
    candidate.checkpoints["latents"][1][0] += 0.1
    reference.decision_trace = ["cache:miss", "cache:hit"]
    candidate.decision_trace = ["cache:miss", "cache:miss"]
    gate = TrajectoryGate(
        ("latents", "hidden"),
        {"latents": {"max_abs": 1e-5, "cosine_similarity": 0.99999}},
    )
    result = evaluate_trajectory_gate(reference, candidate, gate)
    assert not result.passed
    assert "tensor:latents:1:max_abs" in result.failures
    assert "missing_checkpoint:hidden" in result.failures
    assert "decision_trace" in result.failures


def test_gate_keeps_task_metric_separate_from_tensor_parity():
    reference = TrajectoryCapture(outputs={"quality": 1.0})
    candidate = TrajectoryCapture(outputs={"quality": 0.7})
    reference.record("latents", torch.ones(2))
    candidate.record("latents", torch.ones(2))
    gate = TrajectoryGate(
        ("latents",),
        {"latents": {"max_abs": 0.0}},
        {"quality": 0.9},
    )
    result = evaluate_trajectory_gate(
        reference,
        candidate,
        gate,
        output_metric_adapters={"quality": lambda lhs, rhs: 1 - abs(lhs - rhs)},
    )
    assert result.checkpoint_metrics["latents"][0]["max_abs"] == 0.0
    assert result.output_metrics["quality"] == pytest.approx(0.7)
    assert result.failures == ("output_metric:quality",)


def test_gate_rejects_non_finite_tensor_and_output_metrics():
    reference = TrajectoryCapture(outputs={"quality": 1.0})
    candidate = TrajectoryCapture(outputs={"quality": 1.0})
    reference.record("latents", torch.tensor([1.0]))
    candidate.record("latents", torch.tensor([float("nan")]))
    gate = TrajectoryGate(
        ("latents",),
        {"latents": {"max_abs": 0.0, "cosine_similarity": 1.0}},
        {"quality": 1.0},
    )

    result = evaluate_trajectory_gate(
        reference,
        candidate,
        gate,
        output_metric_adapters={"quality": lambda lhs, rhs: float("nan")},
    )

    assert not result.passed
    assert "tensor:latents:0:max_abs" in result.failures
    assert "tensor:latents:0:cosine_similarity" in result.failures
    assert "output_metric:quality" in result.failures


def test_gate_compares_nested_tensor_terminal_state_structurally():
    reference = TrajectoryCapture(
        terminal_state={"cache": [torch.tensor([1, 2]), {"step": 3}]}
    )
    candidate = TrajectoryCapture(
        terminal_state={"cache": [torch.tensor([1, 2]), {"step": 3}]}
    )
    equal = evaluate_trajectory_gate(reference, candidate, TrajectoryGate((), {}))
    assert equal.passed

    candidate.terminal_state["cache"][0][1] = 4
    changed = evaluate_trajectory_gate(reference, candidate, TrajectoryGate((), {}))
    assert changed.failures == ("terminal_state",)


def test_gate_rejects_unsupported_tensor_metric_without_raising():
    reference, candidate = TrajectoryCapture(), TrajectoryCapture()
    reference.record("latents", torch.ones(1))
    candidate.record("latents", torch.ones(1))

    result = evaluate_trajectory_gate(
        reference,
        candidate,
        TrajectoryGate(("latents",), {"latents": {"unsupported_metric": 0.0}}),
    )

    assert not result.passed
    assert result.failures == ("unsupported_tensor_metric:latents:unsupported_metric",)
