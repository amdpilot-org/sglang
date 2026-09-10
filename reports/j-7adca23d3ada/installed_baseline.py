import json
import time

import torch
import triton

from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd


def reference(q, k, v, b_start_loc, b_seq_len, is_causal):
    outputs = []
    scale = q.shape[-1] ** -0.5
    for start, length in zip(b_start_loc.tolist(), b_seq_len.tolist()):
        start = int(start)
        length = int(length)
        qs = q[start : start + length].transpose(0, 1).float()
        ks = k[start : start + length].transpose(0, 1).float()
        vs = v[start : start + length].transpose(0, 1).float()
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                qs, ks, vs, is_causal=is_causal, scale=scale
            ).transpose(0, 1)
        )
    return torch.cat(outputs, dim=0)


torch.manual_seed(7)
device = "cuda"
lengths = [17, 29]
total = sum(lengths)
heads = 2
dim = 64
q = torch.randn(total, heads, dim, device=device, dtype=torch.bfloat16)
k = torch.randn(total, heads, dim, device=device, dtype=torch.bfloat16)
v = torch.randn(total, heads, dim, device=device, dtype=torch.bfloat16)
out = torch.empty_like(q)
b_start_loc = torch.tensor([0, lengths[0]], device=device, dtype=torch.int32)
b_seq_len = torch.tensor(lengths, device=device, dtype=torch.int32)

first_start = time.monotonic()
context_attention_fwd(
    q,
    k,
    v,
    out,
    b_start_loc,
    b_seq_len,
    max(lengths),
    is_causal=True,
)
torch.cuda.synchronize()
first_elapsed_ms = (time.monotonic() - first_start) * 1000.0

expected = reference(q, k, v, b_start_loc, b_seq_len, True)
max_abs = (out.float() - expected).abs().max().item()
mean_abs = (out.float() - expected).abs().mean().item()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
timings_ms = []
for _ in range(3):
    start_event.record()
    context_attention_fwd(
        q,
        k,
        v,
        out,
        b_start_loc,
        b_seq_len,
        max(lengths),
        is_causal=True,
    )
    end_event.record()
    torch.cuda.synchronize()
    timings_ms.append(start_event.elapsed_time(end_event))

window_error = None
try:
    context_attention_fwd(
        q,
        k,
        v,
        out,
        b_start_loc,
        b_seq_len,
        max(lengths),
        is_causal=True,
        window_size=(4, 4),
        s_aux=torch.zeros(heads, device=device, dtype=torch.bfloat16),
    )
except Exception as error:
    window_error = f"{type(error).__name__}: {error}"

result = {
    "label": "installed-source baseline",
    "not_proof_for_later_checkout_changes": True,
    "python": "/opt/venv/bin/python",
    "sglang_python_path": "/sgl-workspace/sglang/python/sglang/__init__.py",
    "kernel_path": "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/prefill_attention.py",
    "triton_path": triton.__file__,
    "torch_path": torch.__file__,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_architecture": list(torch.cuda.get_device_capability(0)),
    "case": {
        "lengths": lengths,
        "heads": heads,
        "head_dim": dim,
        "dtype": "bfloat16",
        "is_causal": True,
    },
    "reference": "torch.nn.functional.scaled_dot_product_attention per varlen sequence",
    "control_max_abs_error": max_abs,
    "control_mean_abs_error": mean_abs,
    "first_gpu_execution_elapsed_ms": first_elapsed_ms,
    "timing_method": "one cold call synchronized with monotonic wall clock, then 3 CUDA events",
    "cuda_event_timings_ms": timings_ms,
    "window_sink_call": {
        "requested": {"window_size": [4, 4], "s_aux_shape": [heads]},
        "supported": window_error is None,
        "error": window_error,
    },
}
print(json.dumps(result, indent=2))
