import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_kernel,
)


DTYPES = (torch.float32, torch.bfloat16, torch.float16)
BATCH_SIZES = (1, 4)
STEP_COUNTS = (1, 4)
ALLOCATED_STEPS = 8
H, HV, K, V = 4, 8, 16, 16


def independent_reference(A_log, dt_bias, a, b, q, k, v, state, batch, steps):
    group = HV // H
    expected = torch.empty(
        batch, steps, HV, V, K, dtype=state.dtype, device=state.device
    )
    for request_index in range(batch):
        hidden_state = state[request_index].float().transpose(-2, -1)
        for step in range(steps):
            token_index = request_index * steps + step
            query = q[0, token_index].float().repeat_interleave(group, dim=0)
            key = k[0, token_index].float().repeat_interleave(group, dim=0)
            value = v[0, token_index].float()
            gate = -A_log.float().exp() * F.softplus(
                a[0, token_index].float() + dt_bias.float()
            )
            beta = torch.sigmoid(b[0, token_index].float())
            query = query / (query.square().sum(dim=1, keepdim=True) + 1e-6).sqrt()
            key = key / (key.square().sum(dim=1, keepdim=True) + 1e-6).sqrt()
            hidden_state = hidden_state * torch.exp(gate)[:, None, None]
            value = value - (hidden_state * key[:, :, None]).sum(dim=1)
            value = value * beta[:, None]
            hidden_state = hidden_state + key[:, :, None] * value[:, None, :]
            expected[request_index, step] = hidden_state.transpose(-2, -1).to(
                state.dtype
            )
    return expected


def make_case(dtype, batch, steps, seed=34786):
    torch.manual_seed(seed)
    A_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    a = torch.randn(1, batch * steps, HV, dtype=torch.float32)
    b = torch.randn(1, batch * steps, HV, dtype=torch.float32)
    q = torch.randn(1, batch * steps, H, K, dtype=torch.float32)
    k = torch.randn(1, batch * steps, H, K, dtype=torch.float32)
    v = torch.randn(1, batch * steps, HV, V, dtype=torch.float32)
    state = torch.randn(batch, HV, V, K, dtype=torch.float32)
    tensors = [A_log, dt_bias, a, b, q, k, v, state]
    A_log, dt_bias, a, b, q, k, v, state = [
        tensor.to(device="cuda", dtype=dtype if tensor is state else torch.bfloat16)
        for tensor in tensors
    ]
    A_log = A_log.to(dtype=dtype)
    state_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(
        0, batch * steps + 1, steps, dtype=torch.int32, device="cuda"
    )
    return A_log, dt_bias, a, b, q, k, v, state, state_indices, cu_seqlens


def run_case(dtype, batch, steps):
    A_log, dt_bias, a, b, q, k, v, state, state_indices, cu_seqlens = make_case(
        dtype, batch, steps
    )
    expected = independent_reference(
        A_log, dt_bias, a, b, q, k, v, state, batch, steps
    )
    buffer = torch.full(
        (batch + 2, ALLOCATED_STEPS, HV, V, K),
        float("nan"),
        dtype=dtype,
        device="cuda",
    )
    write_indices = torch.arange(
        1, batch + 1, dtype=torch.int32, device="cuda"
    )
    buffer_address = buffer.data_ptr()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=False,
        disable_state_update=True,
        intermediate_states_buffer=buffer,
        intermediate_state_indices=write_indices,
        cache_steps=steps,
    )
    end.record()
    torch.cuda.synchronize()
    actual = buffer[1 : batch + 1, :steps]
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-3)
    return {
        "state_dtype": str(dtype).removeprefix("torch."),
        "batch": batch,
        "steps": steps,
        "elapsed_ms": start.elapsed_time(end),
        "max_abs_diff": difference.max().item(),
        "max_rel_diff": relative.max().item(),
        "guard_slots_untouched": bool(
            torch.isnan(buffer[0]).all() and torch.isnan(buffer[batch + 1]).all()
        ),
        "unwritten_steps_untouched": bool(
            torch.isnan(buffer[1 : batch + 1, steps:]).all()
        ),
        "buffer_address_stable": buffer.data_ptr() == buffer_address,
        "slot_stride_elements": buffer.stride(0),
    }


def record_native_dispatch():
    dispatches = []
    for device_index, cache_entry in (
        fused_sigmoid_gating_delta_rule_update_kernel.device_caches.items()
    ):
        for compiled in cache_entry[0].values():
            metadata = compiled.metadata
            dispatches.append(
                {
                    "device_index": device_index,
                    "backend": metadata.backend_name,
                    "arch": metadata.arch,
                    "kernel": metadata.name,
                    "hash": metadata.hash,
                    "num_warps": metadata.num_warps,
                    "num_stages": metadata.num_stages,
                    "asm_artifacts": list(compiled.asm.keys()),
                }
            )
    return dispatches


def record_unsupported_dtype():
    batch, steps = 1, 1
    A_log, dt_bias, a, b, q, k, v, state, indices, cu_seqlens = make_case(
        torch.float64, batch, steps
    )
    state = state.to(torch.float64)
    buffer = torch.zeros(
        batch, steps, HV, V, K, dtype=torch.float64, device="cuda"
    )
    try:
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            disable_state_update=True,
            intermediate_states_buffer=buffer,
            intermediate_state_indices=indices,
            cache_steps=steps,
        )
    except ValueError as error:
        return {"dtype": "torch.float64", "rejected": True, "error": str(error)}
    return {
        "dtype": "torch.float64",
        "rejected": False,
        "error": "No ValueError was raised",
    }


def main():
    torch.cuda.init()
    for dtype in DTYPES:
        for batch in BATCH_SIZES:
            for steps in STEP_COUNTS:
                run_case(dtype, batch, steps)
    results = [
        run_case(dtype, batch, steps)
        for dtype in DTYPES
        for batch in BATCH_SIZES
        for steps in STEP_COUNTS
    ]
    output = {
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
        },
        "matrix": {
            "dtypes": [str(dtype).removeprefix("torch.") for dtype in DTYPES],
            "batch_sizes": list(BATCH_SIZES),
            "step_counts": list(STEP_COUNTS),
            "allocated_steps": ALLOCATED_STEPS,
            "timing_method": (
                "One CUDA event pair around one post-warmup kernel "
                "invocation per case"
            ),
            "input_representation": (
                "Independent float32 master values converted once to each "
                "state dtype; reference consumes the same dtype-specific tensors"
            ),
        },
        "cases": results,
        "native_dispatch": record_native_dispatch(),
        "unsupported_dtype": record_unsupported_dtype(),
    }
    output_path = Path(__file__).with_name("mirror-gpu-results.json")
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
