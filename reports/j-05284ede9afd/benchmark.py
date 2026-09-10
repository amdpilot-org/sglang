#!/usr/bin/env python3
"""Bounded MI300X benchmark for the reduced DeepSeekV2 MLP block.

The workload uses only locally generated tensors.  It compares three existing
activation dispatches while keeping the two bf16 GEMMs and all inputs unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

# Keep AITER importable, but leave the optional AITER activation dispatch off so
# the default resolves to the normal HIP/sgl_kernel path.
os.environ.setdefault("SGLANG_USE_AITER", "1")
os.environ.setdefault("SGLANG_OPT_USE_AITER_SILU_MUL", "0")

import torch


WORKLOADS = (
    (1024, 2048, 512),
    (4096, 2048, 512),
)


def init_single_process_dist() -> None:
    from sglang.test.layer_ut_utils import init_single_process_dist

    init_single_process_dist(master_port=29632, backend="gloo")


def reference_mlp(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate_up = torch.nn.functional.linear(x.float(), gate_up_weight.float())
    gate, up = gate_up[..., : gate_up.shape[-1] // 2], gate_up[..., gate_up.shape[-1] // 2 :]
    activated = torch.nn.functional.silu(gate) * up
    return torch.nn.functional.linear(activated, down_weight.float())


def timed_forward(module: torch.nn.Module, x: torch.Tensor, repeats: int, calls: int) -> dict:
    samples = []
    for _ in range(repeats):
        for _ in range(3):
            module(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            module(x)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / calls)
    return {
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "raw_samples_ms": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    init_single_process_dist()
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    module = DeepseekV2MLP(
        hidden_size=2048,
        intermediate_size=512,
        hidden_act="silu",
        reduce_results=True,
        prefix="benchmark.mlp",
        tp_rank=0,
        tp_size=1,
    ).to(device=device, dtype=dtype)
    torch.manual_seed(1234)
    with torch.no_grad():
        module.gate_up_proj.weight.copy_(
            torch.randn_like(module.gate_up_proj.weight) * 0.02
        )
        module.down_proj.weight.copy_(torch.randn_like(module.down_proj.weight) * 0.02)

    dispatches = {
        "sgl_kernel_hip": module.act_fn.forward_cuda,
        "torch_native": module.act_fn.forward_native,
        "aiter": module.act_fn.forward_aiter,
    }

    results = []
    torch.cuda.reset_peak_memory_stats()
    for tokens, hidden_size, intermediate_size in WORKLOADS:
        torch.manual_seed(4321)
        with torch.no_grad():
            x = torch.randn(tokens, hidden_size, device=device, dtype=dtype) * 0.1
            reference = reference_mlp(
                x,
                module.gate_up_proj.weight,
                module.down_proj.weight,
            )
        for dispatch_name, dispatch_method in dispatches.items():
            module.act_fn._forward_method = dispatch_method
            with torch.no_grad():
                output = module(x)
                torch.cuda.synchronize()
                max_abs_error = (output.float() - reference).abs().max().item()
                reference_peak = reference.abs().max().item()
                timing = timed_forward(module, x, args.repeats, args.calls)
            accepted = max_abs_error < 0.02
            results.append(
                {
                    "workload": {
                        "tokens": tokens,
                        "hidden_size": hidden_size,
                        "intermediate_size": intermediate_size,
                    },
                    "dispatch": dispatch_name,
                    "dtype": str(dtype),
                    "max_abs_error": max_abs_error,
                    "reference_peak_abs": reference_peak,
                    "accuracy_gate": "max_abs_error < 0.02",
                    "accepted": accepted,
                    **timing,
                }
            )
            print(json.dumps(results[-1], sort_keys=True), flush=True)

    report = {
        "label": "reduced DeepSeekV2MLP existing-dispatch benchmark",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": os.popen("git rev-parse HEAD").read().strip(),
        "python": sys.executable,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "workloads": [
            {"tokens": tokens, "hidden_size": hidden, "intermediate_size": inter}
            for tokens, hidden, inter in WORKLOADS
        ],
        "dispatches": list(dispatches),
        "timing": {
            "warmup_per_repeat": 3,
            "measured_calls_per_repeat": args.calls,
            "repeats": args.repeats,
            "method": "CUDA events around real forwards, synchronized after each repeat",
        },
        "max_torch_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0 if all(result["accepted"] for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
