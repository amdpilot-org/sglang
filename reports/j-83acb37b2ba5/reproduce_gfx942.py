import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch


CACHE_ROOT = Path("/tmp/sglang-cache-j-83acb37b2ba5")
TRITON_CACHE = CACHE_ROOT / "checkout-triton"
JIT_CACHE = CACHE_ROOT / "checkout-jit"
TRITON_CACHE.mkdir(parents=True, exist_ok=True)
JIT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["SGLANG_JIT_CACHE_DIR"] = str(JIT_CACHE)
os.environ["TRITON_CACHE_DIR"] = str(TRITON_CACHE)

from sglang.kernels.ops.moe.ep_moe_kernels import (  # noqa: E402
    silu_and_mul_masked_post_quant_fwd,
)


OUTPUT = Path(__file__).with_name("gfx942-results.json")
DEVICE = torch.device("cuda:0")
EXPERTS = 2
MAX_TOKENS = 7
HIDDEN = 256
GROUP_SIZE = 128
GROUPS = HIDDEN // GROUP_SIZE
BATCHES = (1, 3, 5, 7)


def finite_input(batch_index: int) -> torch.Tensor:
    values = torch.linspace(-8.0, 8.0, EXPERTS * MAX_TOKENS * 2 * HIDDEN)
    values[0] = 0.0
    values[1] = 1e-4
    values[-1] = -1e-4
    values = values.view(EXPERTS, MAX_TOKENS, 2 * HIDDEN)
    values = values + 0.017 * (batch_index + 1)
    assert bool(torch.isfinite(values).all())
    return values.to(device=DEVICE, dtype=torch.bfloat16)


def reference(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_input = input_tensor.detach().cpu()
    gate, up = cpu_input.chunk(2, dim=-1)
    gate = gate.float() * torch.sigmoid(gate.float())
    gate = gate.to(torch.bfloat16)
    activation = (up * gate).float()
    grouped = activation.view(
        *activation.shape[:-1], activation.shape[-1] // GROUP_SIZE, GROUP_SIZE
    )
    scale = grouped.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    quantized = (grouped / scale[..., None]).to(torch.float8_e4m3fn)
    return activation, scale, quantized


def check_case(
    output: torch.Tensor,
    output_scale: torch.Tensor,
    input_tensor: torch.Tensor,
    masked_m: torch.Tensor,
) -> dict:
    results = []
    for expert in range(EXPERTS):
        valid = int(masked_m[expert].detach().cpu())
        actual_output = output[expert, :valid].detach().cpu()
        actual_scale = output_scale[expert, :valid].detach().cpu()
        activation, expected_scale, expected_quantized = reference(
            input_tensor[expert, :valid]
        )
        scale_abs_error = (actual_scale - expected_scale).abs()
        scale_rel_error = scale_abs_error / expected_scale.clamp_min(1e-30)
        expanded_scale = actual_scale.repeat_interleave(GROUP_SIZE, dim=-1)
        dequantized = actual_output.float() * expanded_scale
        dequant_error = (dequantized - activation).abs()
        dequant_bound = expanded_scale * 17.0 + 1e-4
        results.append(
            {
                "expert": expert,
                "valid_tokens": valid,
                "scale_max_abs_error": float(scale_abs_error.max()),
                "scale_max_rel_error": float(scale_rel_error.max()),
                "dequantized_max_abs_error": float(dequant_error.max()),
                "dequantized_max_gate_ratio": float(
                    (dequant_error / dequant_bound).max()
                ),
                "output_bytes_equal_reference": bool(
                    torch.equal(
                        actual_output.view(torch.uint8),
                        expected_quantized.view(
                            *expected_quantized.shape[:-2],
                            HIDDEN,
                        ).view(torch.uint8),
                    )
                ),
                "passed_unchanged_gates": bool(
                    torch.allclose(
                        actual_scale,
                        expected_scale,
                        rtol=5e-3,
                        atol=1e-6,
                    )
                    and bool((dequant_error <= dequant_bound).all())
                ),
            }
        )
    return results


jit_error = None
try:
    from sglang.kernels.ops.attention.dsv4.moe import (
        silu_and_mul_contig_post_quant,
    )

    probe_input = finite_input(0)[:1, :1]
    probe_output = torch.empty(
        (1, 1, HIDDEN), device=DEVICE, dtype=torch.float8_e4m3fn
    )
    probe_scale = torch.empty((1, 1, GROUPS), device=DEVICE, dtype=torch.float32)
    silu_and_mul_contig_post_quant(
        probe_input,
        probe_output,
        probe_scale,
        GROUP_SIZE,
    )
except Exception as error:
    jit_error = {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }

input_storage = finite_input(0)
output_storage = torch.full(
    (EXPERTS, MAX_TOKENS, HIDDEN),
    0x7F,
    device=DEVICE,
    dtype=torch.uint8,
).view(torch.float8_e4m3fn)
scale_storage = torch.full(
    (EXPERTS, MAX_TOKENS, GROUPS),
    float("nan"),
    device=DEVICE,
    dtype=torch.float32,
)

reuse_cases = []
for batch_index, batch_size in enumerate(BATCHES):
    input_storage.copy_(finite_input(batch_index))
    output_storage.view(torch.uint8).fill_(0x7F)
    scale_storage.fill_(float("nan"))
    masked_m = torch.tensor(
        [batch_size, max(1, batch_size - 2)],
        device=DEVICE,
        dtype=torch.int32,
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    silu_and_mul_masked_post_quant_fwd(
        input_storage,
        output_storage,
        scale_storage,
        GROUP_SIZE,
        masked_m,
        scale_ue8m0=False,
    )
    end.record()
    torch.cuda.synchronize()
    checks = check_case(output_storage, scale_storage, input_storage, masked_m)
    reuse_cases.append(
        {
            "batch_size_axis": batch_size,
            "masked_m": masked_m.detach().cpu().tolist(),
            "timing_milliseconds": start.elapsed_time(end),
            "experts": checks,
            "passed": all(case["passed_unchanged_gates"] for case in checks),
        }
    )

alias_storage = finite_input(0)[:1].contiguous()
alias_input = alias_storage.view(1, MAX_TOKENS, 2 * HIDDEN)
alias_output = alias_storage.view(torch.float8_e4m3fn).flatten()[
    : MAX_TOKENS * HIDDEN
].view(1, MAX_TOKENS, HIDDEN)
alias_scale = torch.empty(
    (1, MAX_TOKENS, GROUPS), device=DEVICE, dtype=torch.float32
)
alias_masked_m = torch.tensor([3], device=DEVICE, dtype=torch.int32)
alias_reference = reference(alias_input[0, :3])[0]
alias_start = torch.cuda.Event(enable_timing=True)
alias_end = torch.cuda.Event(enable_timing=True)
alias_start.record()
silu_and_mul_masked_post_quant_fwd(
    alias_input,
    alias_output,
    alias_scale,
    GROUP_SIZE,
    alias_masked_m,
    scale_ue8m0=False,
)
alias_end.record()
torch.cuda.synchronize()
alias_output_valid = alias_output[0, :3].detach().cpu().float()
alias_scale_valid = alias_scale[0, :3].detach().cpu()
alias_expanded_scale = alias_scale_valid.repeat_interleave(GROUP_SIZE, dim=-1)
alias_dequant = alias_output_valid * alias_expanded_scale
alias_error = (alias_dequant - alias_reference).abs()
alias_bound = alias_expanded_scale * 17.0 + 1e-4

native_paths = []
for path in TRITON_CACHE.rglob("*"):
    if path.is_file() and "silu" in path.name.lower():
        native_paths.append(str(path))

result = {
    "label": "mirror checkout gfx942 property investigation",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parents[2]
    ).strip(),
    "gpu": {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "architecture": "gfx942",
        "visible_count": torch.cuda.device_count(),
    },
    "image": {
        "required_local_image_id": "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1",
        "hostname_is_not_image_identity": True,
    },
    "paths": {
        "python_module": __import__("sglang.kernels.ops.moe.ep_moe_kernels", fromlist=["x"]).__file__,
        "triton_source": "/job/sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py",
        "jit_cuda_source": "/job/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh",
        "triton_cache": str(TRITON_CACHE),
        "jit_cache": str(JIT_CACHE),
        "native_paths": native_paths,
    },
    "jit_boundary": {
        "status": "unsupported" if jit_error else "supported",
        "error": jit_error,
    },
    "operation": {
        "kernel": "Triton _silu_and_mul_post_quant_kernel via silu_and_mul_masked_post_quant_fwd",
        "input_dtype": "bfloat16",
        "output_dtype": "float8_e4m3fn",
        "scale_dtype": "float32",
        "group_size": GROUP_SIZE,
        "finite_input": True,
        "batch_sizes": list(BATCHES),
        "timing_method": "one CUDA event pair around each real invocation; no warmup or occupancy loop",
        "first_case_note": "the first event measurement includes one-time Triton compilation; later cases use the cached kernel",
    },
    "numerical_gates": {
        "scale": "torch.allclose(actual, expected, rtol=5e-3, atol=1e-6)",
        "dequantized": "abs(error) <= actual_scale * 17.0 + 1e-4",
    },
    "reuse_cases": reuse_cases,
    "alias_probe": {
        "contract": "input/output aliasing is not documented or promised by this kernel",
        "status": "undocumented; numerical probe passed and was not promoted to a supported contract",
        "masked_m": alias_masked_m.detach().cpu().tolist(),
        "timing_milliseconds": alias_start.elapsed_time(alias_end),
        "dequantized_max_abs_error": float(alias_error.max()),
        "dequantized_max_gate_ratio": float((alias_error / alias_bound).max()),
        "passed_unchanged_gates": bool((alias_error <= alias_bound).all()),
    },
    "command": "PYTHONPATH=/job/sglang/python TRITON_CACHE_DIR=/tmp/sglang-cache-j-83acb37b2ba5/checkout-triton SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-83acb37b2ba5/checkout-jit /opt/venv/bin/python reports/j-83acb37b2ba5/reproduce_gfx942.py",
}
result["reuse_passed"] = all(case["passed"] for case in reuse_cases)
OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
