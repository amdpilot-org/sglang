import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import torch
from torch.nn import functional as F

from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / "benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py"
OUTPUT_PATH = Path(__file__).with_name("gpu-evidence.json")

NUM_EXPERTS = 8
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 64
TOP_K = 2
TOKENS_SEQUENCE = (1, 2, 4)
WARM_CALLS = 5
RTOL = 2e-2
ATOL = 2e-2


def load_benchmark_module():
    if importlib.util.find_spec("flashinfer") is None:
        flashinfer = ModuleType("flashinfer")
        flashinfer_testing = ModuleType("flashinfer.testing")
        flashinfer_testing.bench_gpu_time = lambda *args, **kwargs: None
        flashinfer.testing = flashinfer_testing
        sys.modules["flashinfer"] = flashinfer
        sys.modules["flashinfer.testing"] = flashinfer_testing

    spec = importlib.util.spec_from_file_location("benchmark_torch_compile_fused_moe", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(tokens, dtype):
    x = torch.randn(tokens, HIDDEN_SIZE, device="cuda", dtype=dtype) * 0.1
    w1 = torch.randn(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        device="cuda",
        dtype=dtype,
    ) * 0.03
    w2 = torch.randn(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        device="cuda",
        dtype=dtype,
    ) * 0.03
    gating = torch.randn(tokens, NUM_EXPERTS, device="cuda", dtype=torch.float32)
    return x, w1, w2, gating


def independent_reference(x, w1, w2, gating):
    weights = F.softmax(gating.float(), dim=-1)
    weights, ids = torch.topk(weights, TOP_K, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    x_float = x.float()
    w1_float = w1.float()
    w2_float = w2.float()
    output = torch.zeros(x.shape[0], w2.shape[1], device="cuda", dtype=torch.float32)
    for token in range(x.shape[0]):
        for selection in range(TOP_K):
            expert = int(ids[token, selection])
            gate = x_float[token] @ w1_float[expert, :INTERMEDIATE_SIZE].t()
            up = x_float[token] @ w1_float[expert, INTERMEDIATE_SIZE:].t()
            output[token] += weights[token, selection] * (
                F.silu(gate) * up @ w2_float[expert].t()
            )
    return output


def timed_call(function, case, use_fp8=False):
    x, w1, w2, gating = case
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = function(x, w1, w2, gating, TOP_K, use_fp8)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, output


def profile_call(function, case):
    x, w1, w2, gating = case
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        function(x, w1, w2, gating, TOP_K)
        torch.cuda.synchronize()
    return sorted(
        {
            event.key
            for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This bounded investigation requires exactly one CUDA/ROCm device")

    torch.cuda.set_device(0)
    torch.manual_seed(36395)
    set_torch_compile_config()
    benchmark_module = load_benchmark_module()
    compiled_moe = benchmark_module.fused_moe_torch

    device = torch.cuda.get_device_properties(0)
    evidence = {
        "label": "delivery-checkout gfx942 compiled fused-MoE reuse evidence",
        "source": {
            "benchmark_path": str(BENCHMARK_PATH),
            "sglang_path": str(ROOT / "python" / "sglang"),
            "torch_path": str(Path(torch.__file__).resolve()),
            "triton_path": str(Path(__import__("triton").__file__).resolve()),
        },
        "gpu": {
            "name": device.name,
            "compute_capability": [device.major, device.minor],
            "uuid": str(device.uuid),
            "total_memory_bytes": device.total_memory,
        },
        "contracts": {
            "shape_sequence": list(TOKENS_SEQUENCE),
            "dtypes": ["torch.bfloat16", "torch.float16"],
            "warm_calls": WARM_CALLS,
            "reference": "independent float32 per-token/per-expert PyTorch loop",
            "reference_tolerance": {"rtol": RTOL, "atol": ATOL},
            "sentinel_values": [-12345.0, -54321.0],
            "aliasing": "inputs must remain byte-identical; compiled path returns a new output",
            "static_graph_address": "input data pointers are recorded; dynamic=False specializes shapes and dtypes",
            "dtype": "output dtype must match input dtype",
            "unsupported_variant": "use_fp8_w8a8=True must raise AssertionError and must not be forced through",
        },
        "cache_paths": {
            name: os.environ.get(name)
            for name in (
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "SGLANG_JIT_CACHE_DIR",
                "TVM_FFI_CACHE_DIR",
            )
        },
        "timing_method": {
            "cold": "one synchronized call after torch._dynamo.reset() for each dtype, using the externally configured cache",
            "warm": f"mean of {WARM_CALLS} synchronized calls after the cold shape sequence",
            "clock": "time.perf_counter",
        },
        "cases": [],
        "unsupported_fp8": {},
    }

    for dtype in (torch.bfloat16, torch.float16):
        torch._dynamo.reset()
        cases = {tokens: make_case(tokens, dtype) for tokens in TOKENS_SEQUENCE}
        references = {
            tokens: independent_reference(*cases[tokens]) for tokens in TOKENS_SEQUENCE
        }
        input_copies = {
            tokens: [tensor.clone() for tensor in cases[tokens]]
            for tokens in TOKENS_SEQUENCE
        }
        input_addresses = {
            tokens: [hex(tensor.data_ptr()) for tensor in cases[tokens]]
            for tokens in TOKENS_SEQUENCE
        }

        for tokens in TOKENS_SEQUENCE:
            guard_before = torch.full((256,), -12345.0, device="cuda", dtype=torch.float32)
            guard_after = torch.full((256,), -54321.0, device="cuda", dtype=torch.float32)
            cold_ms, output = timed_call(compiled_moe, cases[tokens])
            torch.testing.assert_close(output.float(), references[tokens], rtol=RTOL, atol=ATOL)
            assert output.dtype == dtype
            assert all(
                torch.equal(actual, expected)
                for actual, expected in zip(cases[tokens], input_copies[tokens])
            )
            assert torch.equal(guard_before, torch.full_like(guard_before, -12345.0))
            assert torch.equal(guard_after, torch.full_like(guard_after, -54321.0))

            torch.cuda.synchronize()
            warm_start = time.perf_counter()
            for _ in range(WARM_CALLS):
                warm_output = compiled_moe(*cases[tokens], TOP_K)
            torch.cuda.synchronize()
            warm_ms = (time.perf_counter() - warm_start) * 1000.0 / WARM_CALLS
            torch.testing.assert_close(warm_output.float(), references[tokens], rtol=RTOL, atol=ATOL)
            assert warm_output.dtype == dtype
            difference = (output.float() - references[tokens]).abs()
            evidence["cases"].append(
                {
                    "dtype": str(dtype),
                    "tokens": tokens,
                    "cold_ms": cold_ms,
                    "warm_mean_ms": warm_ms,
                    "max_abs_reference_diff": difference.max().item(),
                    "mean_abs_reference_diff": difference.mean().item(),
                    "input_data_ptrs": input_addresses[tokens],
                    "output_data_ptr": hex(output.data_ptr()),
                    "output_dtype": str(output.dtype),
                    "native_cuda_kernels": profile_call(compiled_moe, cases[tokens]),
                }
            )

    unsupported_case = make_case(1, torch.bfloat16)
    try:
        compiled_moe(*unsupported_case, TOP_K, True)
    except AssertionError as error:
        evidence["unsupported_fp8"] = {
            "raised": True,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    else:
        raise RuntimeError("unsupported fp8 fused-MoE variant unexpectedly succeeded")

    OUTPUT_PATH.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
