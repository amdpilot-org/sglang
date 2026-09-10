import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLA_ATOL,
    MLA_RTOL,
    MLAAttentionCase,
    build_mla_attention_fixture,
    expected_mla_fixture_output,
    run_mla_fixture_eager,
)
from sglang.test.layer_ut_utils import init_single_process_dist

BACKEND = "triton"
DTYPE = torch.float16
DEVICE = "cuda"
HIDDEN_SIZE = 64
KV_LORA_RANK = 32
QK_ROPE_HEAD_DIM = 0
NUM_HEADS = 4
TOKENS = 128
INTERMEDIATE_SIZE = 32
PAGE_SIZE = 1
WARM_REPEATS = 3
SEED = 20260910
RMS_EPS = 1e-5


def _sync() -> None:
    torch.cuda.synchronize()


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "passes_unchanged_gate": bool(
            torch.allclose(actual, expected, atol=MLA_ATOL, rtol=MLA_RTOL)
        ),
    }


def _structured_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    random = torch.randn(TOKENS, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
    cases: dict[str, torch.Tensor] = {}
    cases["zeros"] = torch.zeros(TOKENS, HIDDEN_SIZE, dtype=torch.float32)
    tiny = torch.full((TOKENS, HIDDEN_SIZE), 2.0**-14, dtype=torch.float32)
    tiny[1::2] *= -1.0
    cases["tiny_finite"] = tiny
    magnitude = torch.empty(TOKENS, HIDDEN_SIZE, dtype=torch.float32)
    for row in range(TOKENS):
        scale = (2.0**-10) * (4.0 ** (row % 4))
        magnitude[row] = random[row] * scale
    cases["mixed_magnitudes"] = magnitude
    cancellation = random.clone()
    cancellation[1::2] = -cancellation[0::2]
    cases["pairwise_cancellation"] = cancellation
    logits = torch.arange(HIDDEN_SIZE, dtype=torch.float32)
    skewed = torch.softmax(logits / 0.05, dim=0).expand(TOKENS, HIDDEN_SIZE).clone()
    cases["skewed_probability_state"] = skewed * 4.0
    sign_skew = random * 0.1
    sign_skew[:, : HIDDEN_SIZE // 2] *= -10.0
    cases["negative_sign_skew"] = sign_skew
    for name, values in cases.items():
        assert values.shape == (TOKENS, HIDDEN_SIZE)
        assert torch.isfinite(values).all(), name
    return {name: values.to(DEVICE, DTYPE) for name, values in cases.items()}


def _independent_rmsnorm(
    values: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    values_float = values.float()
    normalized = values_float * torch.rsqrt(
        values_float.square().mean(dim=-1, keepdim=True) + epsilon
    )
    return (normalized * weight.float()).to(values.dtype)


def _independent_mlp(
    values: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    values_float = values.float()
    gate_up = F.linear(values_float, gate_up_weight.float())
    intermediate = gate_up.shape[-1] // 2
    activated = F.silu(gate_up[..., :intermediate]) * gate_up[..., intermediate:]
    return F.linear(activated, down_weight.float()).to(values.dtype)


def _timed_actual_block(
    fixture,
    residual: torch.Tensor,
    rmsnorm: RMSNorm,
    mlp: DeepseekV2MLP,
) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    attention_output = run_mla_fixture_eager(fixture)
    normed = rmsnorm(residual + attention_output)
    mlp_output = mlp(normed)
    output = residual + mlp_output
    _sync()
    return output, time.perf_counter() - start


def _module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return None if spec is None else spec.origin


def _class_path(value: Any) -> dict[str, str | None]:
    module_name = type(value).__module__
    return {
        "class": f"{module_name}.{type(value).__name__}",
        "file": getattr(sys.modules.get(module_name), "__file__", None),
    }


def _gpu_identity() -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "rocm-smi",
                "--showproductname",
                "--showuniqueid",
                "--showserial",
                "--showmeminfo",
                "vram",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        output = f"unavailable: {type(exc).__name__}: {exc}"
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "count": torch.cuda.device_count(),
        "total_bytes": properties.total_memory,
        "gfx": "gfx942",
        "rocm_smi": output,
    }


def _source_identity() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        commit = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "path": str(Path(__file__).resolve().parents[2]),
        "commit": commit,
    }


def _paths(fixture, rmsnorm: RMSNorm, mlp: DeepseekV2MLP) -> dict[str, Any]:
    import aiter
    import sglang
    import triton
    from sglang.srt.models import deepseek_v2

    return {
        "python": "/opt/venv/bin/python",
        "torch": torch.__file__,
        "torch_hip": torch.version.hip,
        "sglang": sglang.__file__,
        "deepseek_v2_module": deepseek_v2.__file__,
        "mla_test_kit": __import__(
            "sglang.test.kits.attention_unittest.attention_methods.mla_attention",
            fromlist=["MLA_ATOL"],
        ).__file__,
        "attention_module": _class_path(fixture.actual_module)["class"],
        "attention_module_file": _class_path(fixture.actual_module)["file"],
        "attention_backend": _class_path(fixture.backend)["class"],
        "attention_backend_file": _class_path(fixture.backend)["file"],
        "mlp_module": _class_path(mlp)["class"],
        "mlp_module_file": _class_path(mlp)["file"],
        "mlp_activation_module": _class_path(mlp.act_fn)["class"],
        "mlp_activation_file": _class_path(mlp.act_fn)["file"],
        "rmsnorm_module": _class_path(rmsnorm)["class"],
        "rmsnorm_file": _class_path(rmsnorm)["file"],
        "aiter": aiter.__file__,
        "triton": triton.__file__,
        "sgl_kernel": _module_origin("sgl_kernel"),
    }


def _weight_bytes(modules: list[torch.nn.Module]) -> int:
    return sum(
        value.element_size() * value.numel()
        for module in modules
        for value in module.state_dict().values()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.json"),
    )
    args = parser.parse_args()

    init_single_process_dist(master_port=29643)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    case = MLAAttentionCase(
        name="structured_complete_block_extend",
        backend=BACKEND,
        forward_mode=ForwardMode.EXTEND,
        num_heads=NUM_HEADS,
        page_size=PAGE_SIZE,
        prefix_lens=(0,),
        extend_lens=(TOKENS,),
    )
    fixture = build_mla_attention_fixture(
        None,
        case,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        hidden_size=HIDDEN_SIZE,
        max_context_len=TOKENS,
        dtype=DTYPE,
        device=DEVICE,
        loc_layout="contiguous",
    )

    mlp = DeepseekV2MLP(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act="silu",
        reduce_results=True,
        tp_size=1,
    ).cuda()
    mlp.to(DTYPE)
    with torch.no_grad():
        torch.nn.init.normal_(mlp.gate_up_proj.weight, mean=0.0, std=0.05)
        torch.nn.init.normal_(mlp.down_proj.weight, mean=0.0, std=0.05)
    rmsnorm = RMSNorm(HIDDEN_SIZE, eps=RMS_EPS).cuda()
    rmsnorm.to(DTYPE)
    with torch.no_grad():
        rmsnorm.weight.fill_(1.0)

    inputs = _structured_inputs()
    results: list[dict[str, Any]] = []
    for case_index, (name, residual) in enumerate(inputs.items()):
        torch.cuda.reset_peak_memory_stats()
        fixture.input_hidden = residual
        actual_block, cold_seconds = _timed_actual_block(
            fixture, residual, rmsnorm, mlp
        )
        attention_actual = run_mla_fixture_eager(fixture)
        attention_reference = expected_mla_fixture_output(fixture)
        normed_actual = rmsnorm(residual + attention_actual)
        mlp_actual = mlp(normed_actual)
        mlp_component_reference = _independent_mlp(
            normed_actual,
            mlp.gate_up_proj.weight,
            mlp.down_proj.weight,
        )
        normed_reference = _independent_rmsnorm(
            residual + attention_reference, rmsnorm.weight, RMS_EPS
        )
        mlp_reference = _independent_mlp(
            normed_reference,
            mlp.gate_up_proj.weight,
            mlp.down_proj.weight,
        )
        complete_reference = residual + mlp_reference

        warm_seconds = []
        for _ in range(WARM_REPEATS):
            _, elapsed = _timed_actual_block(fixture, residual, rmsnorm, mlp)
            warm_seconds.append(elapsed)

        attention_error = _error(attention_actual, attention_reference)
        mlp_error = _error(mlp_actual, mlp_component_reference)
        complete_error = _error(actual_block, complete_reference)
        finite = bool(
            torch.isfinite(actual_block).all()
            and torch.isfinite(complete_reference).all()
        )
        result = {
            "case_index": case_index,
            "name": name,
            "shape": list(residual.shape),
            "dtype": str(residual.dtype).removeprefix("torch."),
            "input_min": residual.min().item(),
            "input_max": residual.max().item(),
            "attention_error": attention_error,
            "mlp_component_error": mlp_error,
            "complete_block_error": complete_error,
            "all_finite": finite,
            "cold_seconds": cold_seconds,
            "warm_seconds": warm_seconds,
            "warm_median_seconds": statistics.median(warm_seconds),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        results.append(result)
        print(
            name,
            "attention",
            attention_error,
            "mlp",
            mlp_error,
            "block",
            complete_error,
            "cold",
            cold_seconds,
        )

    attention_weight_bytes = _weight_bytes([fixture.actual_module])
    mlp_weight_bytes = _weight_bytes([mlp])
    all_pass = all(
        item["attention_error"]["passes_unchanged_gate"]
        and item["mlp_component_error"]["passes_unchanged_gate"]
        and item["complete_block_error"]["passes_unchanged_gate"]
        and item["all_finite"]
        for item in results
    )
    record = {
        "label": "structured-input complete reduced DeepSeek-style block probe",
        "scope": {
            "upstream_reference_issue": "sgl-project/sglang issue 16255",
            "mirror_issue": "amdpilot-org/sglang issue 408",
            "distinct_from_prior_case": "MLA chunk partitioning in PR 486",
            "full_model_quality_claimed": False,
            "weights_downloaded": False,
        },
        "configuration": {
            "backend": BACKEND,
            "dtype": str(DTYPE).removeprefix("torch."),
            "device": DEVICE,
            "tokens": TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_heads": NUM_HEADS,
            "kv_lora_rank": KV_LORA_RANK,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "page_size": PAGE_SIZE,
            "rms_eps": RMS_EPS,
            "seed": SEED,
            "warm_repeats_per_case": WARM_REPEATS,
            "workload_case_count": len(results),
            "total_real_block_forwards": len(results) * (WARM_REPEATS + 1),
        },
        "numerical_contract": {
            "atol": MLA_ATOL,
            "rtol": MLA_RTOL,
            "source": "sglang.test.kits.attention_unittest.attention_methods.mla_attention",
            "required_finite": True,
            "all_cases_pass": all_pass,
        },
        "limits": {
            "wall_limit_seconds": 7200,
            "synthetic_weight_limit_bytes": 4 * 1024**3,
            "live_allocation_limit_bytes": 48 * 1024**3,
            "max_workload_cases": 6,
            "gpu_count": 1,
        },
        "weights": {
            "attention_bytes": attention_weight_bytes,
            "mlp_bytes": mlp_weight_bytes,
            "total_bytes": attention_weight_bytes + mlp_weight_bytes,
            "under_4gb": attention_weight_bytes + mlp_weight_bytes < 4 * 1024**3,
        },
        "source": _source_identity(),
        "command": "/opt/venv/bin/python reports/j-16c101d8c24b/structured_block_probe.py",
        "gpu": _gpu_identity(),
        "image": {
            "operator_specified_image": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
            "operator_specified_local_image_id": "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1",
        },
        "paths": _paths(fixture, rmsnorm, mlp),
        "timing_method": "time.perf_counter around each complete block with torch.cuda.synchronize; cold is first call and warm is three repeats",
        "reference_method": {
            "attention": "independent float32 einsum/softmax MLA reference from the existing test kit",
            "rmsnorm": "independent float32 mean-square normalization",
            "mlp": "independent float32 linear/silu/mul/linear formula",
            "complete_block": "residual plus independent MLP of independent RMSNorm of residual plus independent attention output",
        },
        "cases": results,
        "raw_peak_allocated_bytes": max(
            item["peak_allocated_bytes"] for item in results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if not all_pass:
        raise AssertionError("one or more structured-input numerical gates failed")


if __name__ == "__main__":
    main()
