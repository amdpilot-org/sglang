#!/usr/bin/env python3
"""Bounded gfx942 numerical check for separate FP8/FP4 MoE representations."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, torch_moe_stage1, torch_moe_stage2
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils


TOKENS = 4
HIDDEN = 256
INTERMEDIATE = 128
ROUTED_EXPERTS = 8
ROUTED_TOPK = 2
SHARED_ID = ROUTED_EXPERTS


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(delta) / denominator)


def cosine_distance(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.double().flatten()
    expected = expected.double().flatten()
    denominator = (
        torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(expected)
    ).clamp_min(1e-24)
    return float(1 - torch.dot(actual, expected) / denominator)


def quantize_fp8_blocks(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    experts, rows, cols = weight.shape
    blocks = weight.view(
        experts, rows // 128, 128, cols // 128, 128
    ).permute(0, 1, 3, 2, 4)
    quantized, scale = aiter.pertoken_quant(
        blocks.reshape(experts, -1, 128 * 128), quant_dtype=dtypes.fp8
    )
    quantized = quantized.view(
        experts, rows // 128, cols // 128, 128, 128
    ).permute(0, 1, 3, 2, 4)
    scale = scale.view(experts, rows // 128, cols // 128)
    return quantized.reshape(weight.shape).contiguous(), scale


def dequantize_fp8_blocks(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    expanded = scale.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    return weight.float() * expanded


def dequantize_fp4(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    scale_f32 = fp4_utils.e8m0_to_f32(scale).repeat_interleave(32, dim=-1)
    return fp4_utils.mxfp4_to_f32(weight) * scale_f32


def append_shared_slot(
    routed_ids: torch.Tensor,
    routed_weights: torch.Tensor,
    shared_id: int,
    shared_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.cat(
        (
            routed_ids,
            torch.full(
                (routed_ids.shape[0], 1),
                shared_id,
                dtype=routed_ids.dtype,
                device=routed_ids.device,
            ),
        ),
        dim=1,
    )
    weights = torch.cat(
        (
            routed_weights,
            torch.full(
                (routed_weights.shape[0], 1),
                shared_weight,
                dtype=routed_weights.dtype,
                device=routed_weights.device,
            ),
        ),
        dim=1,
    )
    return ids, weights


def routed_torch_path(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    routed_weights: torch.Tensor,
    routed_ids: torch.Tensor,
) -> torch.Tensor:
    intermediate = torch_moe_stage1(
        hidden,
        w1,
        w2,
        routed_weights,
        routed_ids,
        dtype=dtypes.bf16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=None,
        w1_scale=w1_scale,
        doweight=False,
    )
    return torch_moe_stage2(
        intermediate.view(TOKENS, ROUTED_TOPK, INTERMEDIATE),
        w1,
        w2,
        routed_weights,
        routed_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=True,
    )


def shared_native_path(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    shared_weight: torch.Tensor,
) -> torch.Tensor:
    return fused_moe(
        hidden,
        w1,
        w2,
        shared_weight,
        torch.zeros(
            (TOKENS, 1), dtype=torch.int32, device=hidden.device
        ),
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        doweight_stage1=False,
        gate_mode=GateMode.INTERLEAVE.value,
    )


def independent_reference(
    hidden: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w1_scale: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    routed_weights: torch.Tensor,
    routed_ids: torch.Tensor,
    shared_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    routed_w1_deq = dequantize_fp4(routed_w1, routed_w1_scale)
    routed_w2_deq = dequantize_fp4(routed_w2, routed_w2_scale)
    shared_w1_deq = dequantize_fp8_blocks(shared_w1, shared_w1_scale)
    shared_w2_deq = dequantize_fp8_blocks(shared_w2, shared_w2_scale)

    slot_output = torch.zeros(
        (TOKENS, ROUTED_TOPK, HIDDEN),
        dtype=torch.float32,
        device=hidden.device,
    )
    expanded_hidden = hidden.float()[:, None, :].expand(-1, ROUTED_TOPK, -1)
    for expert_id in range(ROUTED_EXPERTS):
        mask = routed_ids == expert_id
        if not mask.any():
            continue
        gate_up = F.linear(expanded_hidden[mask], routed_w1_deq[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up
        slot_output[mask] = F.linear(intermediate, routed_w2_deq[expert_id])
    routed = (slot_output * routed_weights[..., None]).sum(dim=1)

    gate_up = F.linear(hidden.float(), shared_w1_deq[0])
    gate, up = gate_up.chunk(2, dim=-1)
    intermediate = F.silu(gate) * up
    shared = F.linear(intermediate, shared_w2_deq[0]) * shared_weight
    return routed + shared, routed, shared


def capture_attempt(call) -> dict:
    try:
        call()
    except Exception as error:
        return {
            "supported": False,
            "exception_type": type(error).__name__,
            "message": str(error).splitlines()[0],
        }
    return {"supported": True, "exception_type": None, "message": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-probe-only", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "One CUDA/HIP GPU is required"
    assert torch.cuda.device_count() == 1, "Exactly one GPU must be visible"
    assert get_gfx() == "gfx942", f"Expected gfx942, got {get_gfx()}"

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(38700)
    hidden = torch.randn(
        (TOKENS, HIDDEN), generator=generator, device=device, dtype=dtypes.bf16
    )

    routed_dense_w1 = (
        torch.randn(
            (ROUTED_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
            generator=generator,
            device=device,
            dtype=dtypes.bf16,
        )
        * 0.04
    )
    routed_dense_w2 = (
        torch.randn(
            (ROUTED_EXPERTS, HIDDEN, INTERMEDIATE),
            generator=generator,
            device=device,
            dtype=dtypes.bf16,
        )
        * 0.04
    )
    routed_w1, routed_w1_scale = per_1x32_f4_quant(routed_dense_w1)
    routed_w2, routed_w2_scale = per_1x32_f4_quant(routed_dense_w2)
    routed_w1_scale = routed_w1_scale.view(
        ROUTED_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32
    )
    routed_w2_scale = routed_w2_scale.view(
        ROUTED_EXPERTS, HIDDEN, INTERMEDIATE // 32
    )
    routed_w1 = routed_w1.view(
        ROUTED_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2
    )
    routed_w2 = routed_w2.view(ROUTED_EXPERTS, HIDDEN, INTERMEDIATE // 2)

    shared_dense_w1 = (
        torch.randn(
            (1, 2 * INTERMEDIATE, HIDDEN),
            generator=generator,
            device=device,
            dtype=dtypes.bf16,
        )
        * 0.04
    )
    shared_dense_w2 = (
        torch.randn(
            (1, HIDDEN, INTERMEDIATE),
            generator=generator,
            device=device,
            dtype=dtypes.bf16,
        )
        * 0.04
    )
    shared_w1, shared_w1_scale = quantize_fp8_blocks(shared_dense_w1)
    shared_w2, shared_w2_scale = quantize_fp8_blocks(shared_dense_w2)
    shared_w1_native = shuffle_weight(shared_w1, layout=(16, 16))
    shared_w2_native = shuffle_weight(shared_w2, layout=(16, 16))
    shared_w1_scale_native = fp4_utils.e8m0_shuffle(shared_w1_scale)
    shared_w2_scale_native = fp4_utils.e8m0_shuffle(shared_w2_scale)

    routed_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        dtype=torch.int32,
        device=device,
    )
    raw_routed_weights = torch.tensor(
        [[1.0, 3.0], [2.0, 2.0], [1.0, 1.0], [3.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    routed_weights = raw_routed_weights / raw_routed_weights.sum(
        dim=1, keepdim=True
    )
    shared_weight = torch.ones((TOKENS, 1), dtype=torch.float32, device=device)
    all_ids, all_weights = append_shared_slot(
        routed_ids, routed_weights, SHARED_ID, 1.0
    )

    if args.native_probe_only:
        fused_moe(
            hidden,
            routed_w1,
            routed_w2,
            routed_weights,
            routed_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=routed_w1_scale,
            w2_scale=routed_w2_scale,
            doweight_stage1=False,
            gate_mode=GateMode.SEPARATED.value,
        )
        return 0

    shared_pointers = (
        shared_w1.data_ptr(),
        shared_w2.data_ptr(),
        shared_w1_scale.data_ptr(),
        shared_w2_scale.data_ptr(),
    )
    routed_output = routed_torch_path(
        hidden,
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        routed_weights,
        routed_ids,
    )
    shared_output = shared_native_path(
        hidden,
        shared_w1_native,
        shared_w2_native,
        shared_w1_scale_native,
        shared_w2_scale_native,
        shared_weight,
    )
    combined = routed_output.float() + shared_output.float()

    half_shared_weight = shared_weight * 0.5
    half_shared_output = shared_native_path(
        hidden,
        shared_w1_native,
        shared_w2_native,
        shared_w1_scale_native,
        shared_w2_scale_native,
        half_shared_weight,
    )
    observed_shared = 2.0 * (shared_output.float() - half_shared_output.float())

    half_routed_weights = routed_weights * 0.5
    half_routed_output = routed_torch_path(
        hidden,
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        half_routed_weights,
        routed_ids,
    )
    observed_routed = 2.0 * (routed_output.float() - half_routed_output.float())

    expected, expected_routed, expected_shared = independent_reference(
        hidden,
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        shared_w1,
        shared_w2,
        shared_w1_scale,
        shared_w2_scale,
        routed_weights,
        routed_ids,
        shared_weight,
    )

    combined_error = relative_l2(combined, expected)
    routed_error = relative_l2(routed_output, expected_routed)
    shared_error = relative_l2(shared_output, expected_shared)
    routed_weight_error = relative_l2(observed_routed, expected_routed)
    shared_weight_error = relative_l2(observed_shared, expected_shared)
    combined_cosine = cosine_distance(combined, expected)
    assert torch.isfinite(combined).all()
    assert combined_error <= 5.0e-2, combined_error
    assert routed_weight_error <= 5.0e-3, routed_weight_error
    assert shared_weight_error <= 5.0e-2, shared_weight_error
    assert combined_cosine <= 1.0e-3, combined_cosine
    assert torch.equal(all_ids[:, -1], torch.full_like(all_ids[:, -1], SHARED_ID))
    assert torch.equal(all_weights[:, -1], shared_weight[:, 0])
    assert shared_w1.dtype == dtypes.fp8
    assert shared_w2.dtype == dtypes.fp8
    assert routed_w1.dtype == dtypes.fp4x2
    assert routed_w2.dtype == dtypes.fp4x2
    assert shared_pointers == (
        shared_w1.data_ptr(),
        shared_w2.data_ptr(),
        shared_w1_scale.data_ptr(),
        shared_w2_scale.data_ptr(),
    )

    native_probe = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(args.output),
            "--native-probe-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    native_error_lines = [
        line
        for line in native_probe.stderr.splitlines()
        if "not support" in line.lower() or "error:" in line.lower()
    ]
    native_routed_attempt = {
        "supported": native_probe.returncode == 0 and not native_error_lines,
        "returncode": native_probe.returncode,
        "message": native_error_lines[-1] if native_error_lines else None,
    }
    heterogeneous_attempt = capture_attempt(
        lambda: fused_moe(
            hidden,
            routed_w1,
            routed_w2,
            all_weights,
            all_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=routed_w1_scale,
            w2_scale=routed_w2_scale,
            doweight_stage1=False,
            gate_mode=GateMode.SEPARATED.value,
            shared_w1=shared_w1,
            shared_w2=shared_w2,
            shared_w1_scale=shared_w1_scale,
            shared_w2_scale=shared_w2_scale,
            shared_expert_id=SHARED_ID,
        )
    )

    properties = torch.cuda.get_device_properties(0)
    result = {
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "gfx": get_gfx(),
            "capability": list(torch.cuda.get_device_capability(0)),
            "multiprocessor_count": properties.multi_processor_count,
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "torch_path": torch.__file__,
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "aiter_path": aiter.__file__,
        },
        "representation": {
            "routed_weight_dtype": str(routed_w1.dtype),
            "routed_scale_dtype": str(routed_w1_scale.dtype),
            "shared_weight_dtype": str(shared_w1.dtype),
            "shared_scale_dtype": str(shared_w1_scale.dtype),
            "shared_pointers_unchanged": True,
        },
        "mapping": {
            "routed_topk": ROUTED_TOPK,
            "expanded_topk": all_ids.shape[1],
            "shared_slot": SHARED_ID,
            "shared_weight": 1.0,
        },
        "metrics": {
            "combined_relative_l2": combined_error,
            "routed_relative_l2": routed_error,
            "shared_relative_l2": shared_error,
            "combined_cosine_distance": combined_cosine,
            "routed_weight_probe_relative_l2": routed_weight_error,
            "shared_weight_probe_relative_l2": shared_weight_error,
            "max_abs_error": float((combined - expected).abs().max()),
        },
        "kernel_attempts": {
            "native_gfx942_routed_fp4": native_routed_attempt,
            "heterogeneous_fp8_shared_fp4_routed": heterogeneous_attempt,
        },
        "gates": {
            "combined_relative_l2_max": 5.0e-2,
            "routed_weight_probe_relative_l2_max": 5.0e-3,
            "shared_weight_probe_relative_l2_max": 5.0e-2,
            "combined_cosine_distance_max": 1.0e-3,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
