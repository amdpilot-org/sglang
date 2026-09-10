import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/sglang-cache-j-8f721bfd968a/triton")
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", "/tmp/sglang-cache-j-8f721bfd968a/torch"
)

import torch


EXPERTS = 8
TOPK = 2
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 64
TOKEN_COUNTS = (1, 4, 16)
GUARD_SIZE = 32
SENTINEL = -12345.0
WARMUP_CALLS = 2
TIMED_CALLS = 10


def independent_reference(hidden_states, w1, w2, topk_weights, topk_ids):
    token_count = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    expanded = hidden_states.repeat_interleave(topk, dim=0)
    expert_outputs = torch.empty(
        token_count * topk,
        HIDDEN_SIZE,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    for expert_index in range(EXPERTS):
        mask = (topk_ids == expert_index).reshape(-1)
        if not bool(mask.any()):
            continue
        gate_up = expanded[mask] @ w1[expert_index].t()
        gate, up = gate_up.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        expert_outputs[mask] = activated @ w2[expert_index].t()
    return (
        expert_outputs.view(token_count, topk, HIDDEN_SIZE)
        * topk_weights.to(expert_outputs.dtype).unsqueeze(-1)
    ).sum(dim=1)


def make_batch(token_count, dtype, device):
    logits = torch.randn(token_count, EXPERTS, device=device, dtype=dtype)
    probabilities = torch.softmax(logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probabilities, TOPK, dim=-1)
    return topk_weights.to(dtype), topk_ids.to(torch.int32)


def record_dispatch(module, name, calls):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        calls.append(
            {
                "name": name,
                "module": original.__module__,
                "qualified_name": original.__qualname__,
                "arg_count": len(args),
                "kwarg_names": sorted(kwargs),
            }
        )
        return original(*args, **kwargs)

    setattr(module, name, wrapper)


def timed_call(function, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_CALLS):
        function(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED_CALLS


def main():
    overall_started = time.perf_counter()
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe_module
    import sglang.kernels.ops.moe.fused_moe_triton_kernels as kernels
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=(
            f"file:///tmp/sglang-j-8f721bfd968a-{os.getpid()}"
        ),
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    dispatch_calls = []
    for name in (
        "invoke_fused_moe_kernel",
        "act_and_mul_triton",
        "moe_sum_reduce_triton",
    ):
        record_dispatch(kernels, name, dispatch_calls)
        setattr(fused_moe_module, name, getattr(kernels, name))
    for name in ("silu_and_mul", "moe_sum_reduce_torch_compile"):
        record_dispatch(fused_moe_module, name, dispatch_calls)

    torch.manual_seed(32312)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    w1 = torch.randn(
        EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype
    ).mul_(0.02)
    w2 = torch.randn(
        EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype
    ).mul_(0.02)

    max_token_count = max(TOKEN_COUNTS)
    guarded_storage = torch.full(
        (GUARD_SIZE + max_token_count * HIDDEN_SIZE + GUARD_SIZE,),
        SENTINEL,
        device=device,
        dtype=dtype,
    )
    before_guard = guarded_storage[:GUARD_SIZE].clone()
    after_guard = guarded_storage[-GUARD_SIZE:].clone()
    reused_view = guarded_storage[
        GUARD_SIZE : GUARD_SIZE + max_token_count * HIDDEN_SIZE
    ].view(max_token_count, HIDDEN_SIZE)
    reused_address = int(reused_view.data_ptr())

    batches = []
    for token_count in TOKEN_COUNTS:
        hidden = torch.randn(
            token_count, HIDDEN_SIZE, device=device, dtype=dtype
        ).mul_(0.05)
        topk_weights, topk_ids = make_batch(token_count, dtype, device)
        batches.append((hidden, topk_weights, topk_ids))

    first_call_started = time.perf_counter()
    fused_moe_module.fused_moe(
        batches[0][0],
        w1,
        w2,
        (batches[0][1], batches[0][2], None),
        MoeRunnerConfig(inplace=False),
    )
    torch.cuda.synchronize()
    first_gpu_execution_elapsed = time.perf_counter() - first_call_started

    numerical_results = []
    reuse_addresses = []
    for mode in ("fresh", "reuse"):
        for batch_index, (hidden, topk_weights, topk_ids) in enumerate(batches):
            reference = independent_reference(
                hidden, w1, w2, topk_weights, topk_ids
            )
            input_sentinel = hidden.clone()
            if mode == "fresh":
                output = fused_moe_module.fused_moe(
                    hidden,
                    w1,
                    w2,
                    (topk_weights, topk_ids, None),
                    MoeRunnerConfig(inplace=False),
                )
                output_address = int(output.data_ptr())
                input_address = int(hidden.data_ptr())
                input_unchanged = bool(torch.equal(hidden, input_sentinel))
            else:
                output = reused_view[: hidden.shape[0]]
                output.copy_(hidden)
                fused_moe_module.fused_moe(
                    output,
                    w1,
                    w2,
                    (topk_weights, topk_ids, None),
                    MoeRunnerConfig(inplace=True),
                )
                output_address = int(output.data_ptr())
                input_address = int(output.data_ptr())
                reuse_addresses.append(output_address)
                input_unchanged = False
            torch.cuda.synchronize()
            difference = (output.float() - reference.float()).abs()
            numerical_results.append(
                {
                    "mode": mode,
                    "batch_index": batch_index,
                    "token_count": hidden.shape[0],
                    "input_address": input_address,
                    "output_address": output_address,
                    "input_unchanged": input_unchanged,
                    "output_dtype": str(output.dtype),
                    "output_shape": list(output.shape),
                    "guards_unchanged": bool(
                        torch.equal(guarded_storage[:GUARD_SIZE], before_guard)
                        and torch.equal(guarded_storage[-GUARD_SIZE:], after_guard)
                    ),
                    "max_abs_difference": float(difference.max().item()),
                    "mean_abs_difference": float(difference.mean().item()),
                    "allclose_rtol_2e_2_atol_2e_2": bool(
                        torch.allclose(
                            output.float(),
                            reference.float(),
                            rtol=2e-2,
                            atol=2e-2,
                        )
                    ),
                }
            )

    unsupported = []
    try:
        fused_moe_module.fused_moe(
            batches[0][0].double(),
            w1.double(),
            w2.double(),
            (batches[0][1], batches[0][2], None),
            MoeRunnerConfig(inplace=False),
        )
    except Exception as error:
        unsupported.append(
            {
                "variant": "float64",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    try:
        fused_moe_module.fused_moe(
            batches[0][0],
            w1,
            w2,
            (batches[0][1], batches[0][2], None),
            MoeRunnerConfig(inplace=True, no_combine=True),
        )
    except Exception as error:
        unsupported.append(
            {
                "variant": "inplace + no_combine",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    timing_matrix = []
    for mode in ("fresh", "reuse"):
        for batch_index, (hidden, topk_weights, topk_ids) in enumerate(batches):
            config = MoeRunnerConfig(inplace=(mode == "reuse"))
            input_tensor = (
                reused_view[: hidden.shape[0]] if mode == "reuse" else hidden
            )
            if mode == "reuse":
                input_tensor.copy_(hidden)
            for _ in range(WARMUP_CALLS):
                fused_moe_module.fused_moe(
                    input_tensor,
                    w1,
                    w2,
                    (topk_weights, topk_ids, None),
                    config,
                )
            mean_ms = timed_call(
                fused_moe_module.fused_moe,
                input_tensor,
                w1,
                w2,
                (topk_weights, topk_ids, None),
                config,
            )
            timing_matrix.append(
                {
                    "mode": mode,
                    "batch_index": batch_index,
                    "token_count": hidden.shape[0],
                    "warmup_calls": WARMUP_CALLS,
                    "timed_calls": TIMED_CALLS,
                    "mean_milliseconds": mean_ms,
                }
            )

    gpu = torch.cuda.get_device_properties(0)
    report = {
        "label": "persistent mirror checkout experiment",
        "campaign": "repo-e2e-20260909",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_base": subprocess.check_output(
            ["git", "rev-parse", "main"], text=True
        ).strip(),
        "captured_at_utc": subprocess.check_output(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True
        ).strip(),
        "first_gpu_execution_elapsed_seconds": first_gpu_execution_elapsed,
        "total_elapsed_seconds": time.perf_counter() - overall_started,
        "environment": {
            "python": "/opt/venv/bin/python",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_path": torch.__file__,
            "triton_version": __import__("triton").__version__,
            "triton_path": __import__("triton").__file__,
            "sglang_path": __import__("sglang").__file__,
            "fused_moe_path": fused_moe_module.__file__,
            "kernel_path": kernels.__file__,
            "sgl_kernel_path": __import__("sgl_kernel").__file__,
            "gpu_name": gpu.name,
            "gpu_uuid": str(gpu.uuid),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_memory_bytes": gpu.total_memory,
            "image": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
            "image_id": (
                "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c"
                "cd4dfcc1147ff1"
            ),
        },
        "shape": {
            "experts": EXPERTS,
            "topk": TOPK,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "dtype": str(dtype),
            "token_counts": list(TOKEN_COUNTS),
        },
        "reference": {
            "method": "independent per-expert torch.matmul + silu * up + weighted sum",
            "comparison_dtype": "float32",
            "gate": "torch.allclose rtol=2e-2 atol=2e-2",
        },
        "sentinel": {
            "method": "32-element before/after guards around reused input/output storage",
            "reuse_address": reused_address,
            "reuse_addresses": reuse_addresses,
            "reuse_address_stable": len(set(reuse_addresses)) == 1,
            "fresh_output_distinct_from_input": all(
                result["output_address"] != result["input_address"]
                for result in numerical_results
                if result["mode"] == "fresh"
            ),
            "reuse_output_same_as_input": all(
                result["output_address"] == result["input_address"]
                for result in numerical_results
                if result["mode"] == "reuse"
            ),
            "guards_unchanged": all(
                result["guards_unchanged"] for result in numerical_results
            ),
        },
        "numerical_results": numerical_results,
        "unsupported_variants": unsupported,
        "timing_method": (
            f"{WARMUP_CALLS} warmup calls, then {TIMED_CALLS} calls between "
            "CUDA events with synchronization"
        ),
        "timing_matrix": timing_matrix,
        "native_dispatch_summary": {
            name: sum(call["name"] == name for call in dispatch_calls)
            for name in sorted({call["name"] for call in dispatch_calls})
        },
        "native_dispatch_sample": dispatch_calls[:12],
        "commands": [
            (
                "PYTHONPATH=/job/sglang/python TRITON_CACHE_DIR=/tmp/sglang-cache-"
                "j-8f721bfd968a/triton /opt/venv/bin/python reports/j-8f721bfd968a/"
                "run_reuse_experiment.py"
            ),
            (
                "PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q "
                "test/registered/unit/layers/moe/test_fused_moe_output_reuse.py"
            ),
        ],
    }
    output_path = Path(__file__).with_name("results.json")
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
