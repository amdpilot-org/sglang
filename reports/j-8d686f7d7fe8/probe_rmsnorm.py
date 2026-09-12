#!/usr/bin/env python3
"""Focused one-GPU numerical qualification for SGLang RMSNorm on ROCm."""

import inspect
import json
import os
import sys

import torch
import sglang.srt.layers.layernorm as layernorm_module
from sglang.srt.layers.layernorm import RMSNorm


def reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Independent float32 RMSNorm reference, without calling RMSNorm APIs."""
    x32 = x.float()
    weight32 = weight.float()
    inverse_rms = torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    return x32 * inverse_rms * weight32


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one assigned GPU, got {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(0)
    torch.manual_seed(20260912)
    eps = 1e-6
    tolerances = {
        torch.float16: {"atol": 0.004, "rtol": 0.004},
        torch.bfloat16: {"atol": 0.025, "rtol": 0.01},
    }
    cases = []

    for dtype in (torch.float16, torch.bfloat16):
        for hidden_width in (4096, 4103):
            # Nontrivial deterministic values avoid a random-only smoke test.
            x = torch.randn((7, hidden_width), device="cuda", dtype=dtype) * 1.7
            weight = (
                torch.linspace(0.25, 1.75, hidden_width, device="cuda")
                .sin()
                .add(1.0)
                .to(dtype)
            )
            op = RMSNorm(hidden_width, eps=eps, weight_dtype=dtype).cuda()
            with torch.no_grad():
                op.weight.copy_(weight)
                actual = op(x)
            torch.cuda.synchronize()

            dispatch = op._forward_method.__name__
            if dispatch != "forward_hip":
                raise AssertionError(f"unexpected RMSNorm dispatch: {dispatch}")

            expected = reference(x, weight, eps)
            actual32 = actual.float()
            abs_error = (actual32 - expected).abs()
            rel_error = abs_error / expected.abs().clamp_min(1e-7)
            atol = tolerances[dtype]["atol"]
            rtol = tolerances[dtype]["rtol"]
            passed = torch.allclose(actual32, expected, atol=atol, rtol=rtol)
            case = {
                "dtype": str(dtype).removeprefix("torch."),
                "shape": list(x.shape),
                "hidden_width_is_odd": bool(hidden_width % 2),
                "dispatch": dispatch,
                "output_dtype": str(actual.dtype).removeprefix("torch."),
                "atol": atol,
                "rtol": rtol,
                "max_absolute_error": abs_error.max().item(),
                "max_relative_error": rel_error.max().item(),
                "allclose": passed,
                "finite": bool(torch.isfinite(actual).all().item()),
            }
            cases.append(case)
            print("CASE " + json.dumps(case, sort_keys=True))
            if not passed or not case["finite"]:
                raise AssertionError(f"numerical check failed: {case}")

    props = torch.cuda.get_device_properties(0)
    summary = {
        "status": "pass",
        "python": sys.executable,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "visible_device_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            )
        },
        "gpu": {
            "logical_index": 0,
            "visible_device_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "architecture": props.gcnArchName,
            "pci_bus_id": props.pci_bus_id,
            "uuid": str(props.uuid),
            "total_memory_bytes": props.total_memory,
        },
        "imports": {
            "sglang_layernorm_source": inspect.getsourcefile(layernorm_module.RMSNorm),
            "sglang_layernorm_module": layernorm_module.__file__,
            "torch_module": torch.__file__,
            "torch_native_extension": torch._C.__file__,
        },
        "backend_flags": {
            "is_hip": layernorm_module._is_hip,
            "use_aiter": layernorm_module._use_aiter,
            "has_vllm_rms_norm": layernorm_module._has_vllm_rms_norm,
            "effective_path": (
                "forward_hip -> forward_native"
                if not layernorm_module._has_vllm_rms_norm
                else "forward_hip -> vllm._custom_ops.rms_norm"
            ),
        },
        "epsilon": eps,
        "reference": "float32 x * rsqrt(mean(x*x, dim=-1) + eps) * float32 weight",
        "cases": cases,
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
