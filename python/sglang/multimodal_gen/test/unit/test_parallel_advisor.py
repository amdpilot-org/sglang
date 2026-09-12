import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.server_args.parallel_advisor import (
    CalibrationReport,
    CalibrationResult,
    DiffusionWorkloadSignature,
    ParallelExecutionPlan,
    PlanCapabilities,
    apply_advisor_to_server_args,
    enumerate_legal_plans,
    resolve_calibrated_plan,
    validate_plan,
)


@pytest.fixture
def signature():
    return DiffusionWorkloadSignature(
        model_revision="model@abc",
        task_type="t2v",
        latent_tokens=128,
        num_inference_steps=20,
        logical_batch=1,
        candidate_count=1,
        cfg_branches=2,
        dtype="bf16",
        cache_mode="none",
        compile_mode="eager",
        device_topology="2xMI300X/xGMI",
    )


def result(signature, plan, p95=10, quality="pass", feasible=True):
    return CalibrationResult(
        signature=signature,
        plan=plan,
        feasible=feasible,
        latency_ms={"p50": p95 - 1, "p95": p95},
        peak_memory_bytes=1024,
        quality_gate=quality,
        communication_summary={"collective_count": 1, "payload_bytes": 8},
    )


def test_enumeration_rejects_illegal_products_and_divisibility(signature):
    caps = PlanCapabilities(
        num_devices=4,
        cfg_parallel_sizes=(1, 2),
        tensor_parallel_sizes=(1, 2, 4),
        sequence_parallel_sizes=(1, 3, 4),
        data_parallel_replicas=(1, 2, 4),
        supports_fsdp=True,
        attention_heads=6,
    )
    plans = enumerate_legal_plans(signature, caps)
    assert plans
    assert all(plan.required_devices == 4 for plan in plans)
    assert all(6 % plan.tensor_parallel_size == 0 for plan in plans)
    assert all(128 % plan.sequence_parallel_size == 0 for plan in plans)
    assert all(plan.cfg_parallel_size <= 2 for plan in plans)


def test_memory_and_cfg_gates_are_explanatory(signature):
    plan = ParallelExecutionPlan(2, cfg_parallel_size=2)
    caps = PlanCapabilities(
        num_devices=2,
        cfg_parallel_sizes=(1, 2),
        available_memory_bytes=100,
    )
    constrained = DiffusionWorkloadSignature(**{**asdict(signature), "cfg_branches": 1})
    reasons = validate_plan(plan, constrained, caps, estimated_peak_memory_bytes=101)
    assert "CFG degree exceeds workload branches" in reasons
    assert "estimated peak memory exceeds least-free device" in reasons


def test_exact_match_ranks_only_feasible_quality_passing_results(signature):
    slow = ParallelExecutionPlan(2, cfg_parallel_size=2)
    fast_but_bad = ParallelExecutionPlan(2, tensor_parallel_size=2)
    report = CalibrationReport(
        code_revision="code1",
        environment={},
        results=(
            result(signature, slow, 20),
            result(signature, fast_but_bad, 5, quality="failed: trajectory"),
        ),
    )
    resolution = resolve_calibrated_plan(report, signature, code_revision="code1")
    assert resolution.plan == slow
    assert resolution.source == "exact_calibration"
    assert "quality gate=failed: trajectory" in resolution.reasons


@pytest.mark.parametrize(
    "changed",
    [
        {"candidate_count": 2},
        {"num_inference_steps": 21},
        {"model_revision": "model@new"},
        {"device_topology": "2xMI300X/PCIe"},
    ],
)
def test_incompatible_signature_fails_closed(signature, changed):
    report = CalibrationReport(
        "code1", {}, (result(signature, ParallelExecutionPlan(1)),)
    )
    other = DiffusionWorkloadSignature(**{**asdict(signature), **changed})
    resolution = resolve_calibrated_plan(report, other)
    assert resolution.plan is None
    assert resolution.reasons == ("no exact signature match",)


def test_stale_code_revision_fails_closed(signature):
    report = CalibrationReport(
        "old", {}, (result(signature, ParallelExecutionPlan(1)),)
    )
    assert resolve_calibrated_plan(report, signature, code_revision="new").plan is None


def test_stale_environment_fails_closed(signature):
    report = CalibrationReport(
        "code1",
        {"backend": "nccl", "device": "MI300X", "driver": "6.3"},
        (result(signature, ParallelExecutionPlan(1)),),
    )
    resolution = resolve_calibrated_plan(
        report,
        signature,
        code_revision="code1",
        environment={"backend": "rccl", "device": "MI355X", "driver": "7.2"},
    )
    assert resolution.plan is None
    assert resolution.reasons == ("calibration environment mismatch",)


def test_report_only_stale_driver_version_fails_closed(signature):
    report = CalibrationReport(
        "code1",
        {"device": "MI350X", "driver_version": "stale"},
        (result(signature, ParallelExecutionPlan(1)),),
    )
    resolution = resolve_calibrated_plan(
        report,
        signature,
        code_revision="code1",
        environment={"device": "MI350X"},
    )
    assert resolution.plan is None
    assert resolution.reasons == ("calibration environment mismatch",)


def test_feasible_report_cannot_select_internally_illegal_plan(signature):
    illegal = ParallelExecutionPlan(
        num_gpus=1,
        cfg_parallel_size=2,
        tensor_parallel_size=2,
        sequence_parallel_size=2,
        fsdp=True,
        data_parallel_replicas=2,
    )
    report = CalibrationReport("code1", {}, (result(signature, illegal),))
    resolution = resolve_calibrated_plan(report, signature)
    assert resolution.plan is None
    assert resolution.reasons == ("parallel degree product does not equal num_gpus",)


def test_feasible_report_cannot_select_cfg_plan_for_one_branch(signature):
    one_branch = DiffusionWorkloadSignature(**{**asdict(signature), "cfg_branches": 1})
    plan = ParallelExecutionPlan(num_gpus=2, cfg_parallel_size=2)
    report = CalibrationReport("code1", {}, (result(one_branch, plan),))
    resolution = resolve_calibrated_plan(report, one_branch)
    assert resolution.plan is None
    assert resolution.reasons == ("CFG degree exceeds workload branches",)


def test_report_round_trip_and_stable_hash(tmp_path, signature):
    report = CalibrationReport(
        "code", {"driver": "x"}, (result(signature, ParallelExecutionPlan(1)),)
    )
    path = tmp_path / "report.json"
    report.dump(path)
    loaded = CalibrationReport.load(path)
    assert loaded == report
    assert loaded.results[0].signature.stable_hash == signature.stable_hash


def test_explicit_flags_override_report(tmp_path, signature):
    args = SimpleNamespace(
        diffusion_parallel_plan="auto:missing.json",
        diffusion_workload_signature=json.dumps(asdict(signature)),
        _explicit_arg_names={"tp_size"},
    )
    resolution = apply_advisor_to_server_args(args)
    assert resolution.source == "explicit_flags"


def test_advisor_applies_exact_world_size_one_plan(tmp_path, signature, monkeypatch):
    plan = ParallelExecutionPlan(1)
    environment = {"backend": "rccl", "device": "MI355X", "driver": "7.2"}
    report = CalibrationReport("code", environment, (result(signature, plan),))
    path = tmp_path / "report.json"
    report.dump(path)
    args = SimpleNamespace(
        diffusion_parallel_plan=f"auto:{path}",
        diffusion_workload_signature=json.dumps(asdict(signature)),
        _explicit_arg_names=set(),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.parallel_advisor.get_git_commit_hash",
        lambda: "code",
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.parallel_advisor.get_runtime_environment",
        lambda: environment,
    )
    resolution = apply_advisor_to_server_args(args)
    assert resolution.plan == plan
    assert resolution.signature_hash == signature.stable_hash
    assert args.num_gpus == args.tp_size == args.sp_degree == 1
    assert args.cfg_parallel_degree == args.dp_size == 1
    assert args.enable_cfg_parallel is False
    assert args.use_fsdp_inference is False


@pytest.mark.parametrize(
    ("recorded_code", "recorded_environment", "expected_reason"),
    [
        ("old-code", {"device": "current-device"}, "code revision mismatch"),
        (
            "current-code",
            {"device": "old-device"},
            "calibration environment mismatch",
        ),
    ],
)
def test_startup_rejects_stale_revision_and_environment(
    tmp_path,
    signature,
    monkeypatch,
    recorded_code,
    recorded_environment,
    expected_reason,
):
    plan = ParallelExecutionPlan(1)
    report = CalibrationReport(
        recorded_code, recorded_environment, (result(signature, plan),)
    )
    path = tmp_path / "report.json"
    report.dump(path)
    args = SimpleNamespace(
        diffusion_parallel_plan=f"auto:{path}",
        diffusion_workload_signature=json.dumps(asdict(signature)),
        _explicit_arg_names=set(),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.parallel_advisor.get_git_commit_hash",
        lambda: "current-code",
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.parallel_advisor.get_runtime_environment",
        lambda: {"device": "current-device"},
    )
    resolution = apply_advisor_to_server_args(args)
    assert resolution.plan is None
    assert resolution.reasons == (expected_reason,)
