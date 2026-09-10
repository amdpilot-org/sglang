import json
import math
import subprocess
import time

import torch
import triton

from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd
from sglang.srt.layers.attention.vision import VisionTritonAttention


def explicit_mask_reference(q, k, v, starts, lengths, scale, window, sinks):
    outputs = []
    for start, length in zip(starts, lengths):
        start = int(start)
        length = int(length)
        query = q[start : start + length].transpose(0, 1).float()
        key = k[start : start + length].transpose(0, 1).float()
        value = v[start : start + length].transpose(0, 1).float()
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        query_index = torch.arange(length, device=q.device)[:, None]
        key_index = torch.arange(length, device=q.device)[None, :]
        allowed = torch.ones(
            (length, length), dtype=torch.bool, device=q.device
        )
        left, right = window
        relative = key_index - query_index
        if left >= 0:
            allowed &= relative >= -left
        if right >= 0:
            allowed &= relative <= right
        scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))
        if sinks is not None:
            scores = torch.cat(
                [scores, sinks.float()[:, None, None].expand(-1, length, 1)], dim=-1
            )
        probabilities = torch.softmax(scores, dim=-1)
        if sinks is not None:
            probabilities = probabilities[..., :-1]
        outputs.append(torch.matmul(probabilities, value).transpose(0, 1))
    return torch.cat(outputs, dim=0)


torch.manual_seed(17)
device = "cuda"
lengths = [37, 53]
starts = [0, lengths[0]]
total = sum(lengths)
heads = 4
head_dim = 64
scale = head_dim**-0.5
q = torch.randn(total, heads, head_dim, device=device, dtype=torch.bfloat16)
k = torch.randn(total, heads, head_dim, device=device, dtype=torch.bfloat16)
v = torch.randn(total, heads, head_dim, device=device, dtype=torch.bfloat16)
cu_seqlens = torch.tensor([0, *starts[1:], total], device=device, dtype=torch.int32)
sequence_lengths = torch.tensor(lengths, device=device, dtype=torch.int32)
sinks = torch.randn(heads, device=device, dtype=torch.bfloat16)
backend = VisionTritonAttention(use_data_parallel=True)


def run_backend(window, sink_values):
    return backend(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        bsz=len(lengths),
        seq_len=max(lengths),
        softmax_scale=scale,
        sequence_lengths=sequence_lengths,
        max_seqlen=max(lengths),
        window_size=window,
        s_aux=sink_values,
    )


cases = [
    ("full_attention", (-1, -1), None),
    ("local_window", (5, 7), None),
    ("sink_token", (-1, -1), sinks),
    ("local_window_and_sink", (5, 7), sinks),
]

first_start = time.monotonic()
first_output = run_backend((5, 7), sinks)
torch.cuda.synchronize()
first_elapsed_ms = (time.monotonic() - first_start) * 1000.0

results = []
for name, window, sink_values in cases:
    output = run_backend(window, sink_values)
    reference = explicit_mask_reference(
        q, k, v, starts, lengths, scale, window, sink_values
    )
    difference = (output.float() - reference).abs()
    max_abs = difference.max().item()
    mean_abs = difference.mean().item()
    passed = torch.allclose(output.float(), reference, atol=0.02, rtol=0.02)
    results.append(
        {
            "case": name,
            "window_size": list(window),
            "sinks": None if sink_values is None else "random per-head logits",
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "atol": 0.02,
            "rtol": 0.02,
            "passed": bool(passed),
        }
    )

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
timings_ms = []
for _ in range(3):
    start_event.record()
    run_backend((5, 7), sinks)
    end_event.record()
    torch.cuda.synchronize()
    timings_ms.append(start_event.elapsed_time(end_event))

gpu_json = subprocess.check_output(
    ["rocm-smi", "--showproductname", "--showgpu", "--showserial", "--json"],
    text=True,
)

result = {
    "label": "upstream PR 38142 candidate on persistent mirror checkout",
    "candidate_commit": "7284c6da0d08a5bfa0d241d08c8ceb8e8738e55e",
    "mirror_main_base_commit": "0084030179bfba86bfeb6d43f7997d4076329d2c",
    "python": "/opt/venv/bin/python",
    "sglang_python_path": "/job/sglang/python/sglang/__init__.py",
    "vision_backend_path": "/job/sglang/python/sglang/srt/layers/attention/vision.py",
    "kernel_path": "/job/sglang/python/sglang/kernels/ops/attention/prefill_attention.py",
    "native_module_kind": "Triton JIT kernel; no prebuilt .so is loaded by this path",
    "triton_path": triton.__file__,
    "torch_path": torch.__file__,
    "gpu": {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_hip_version": torch.version.hip,
        "rocm_smi_json": gpu_json.strip(),
    },
    "case_shape": {
        "lengths": lengths,
        "heads": heads,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "backend": "VisionTritonAttention(use_data_parallel=True)",
    },
    "reference": "independent float32 explicit boolean mask plus virtual sink-logit softmax",
    "results": results,
    "first_gpu_execution_elapsed_ms": first_elapsed_ms,
    "timing_method": "one synchronized cold call with monotonic wall clock, then 3 CUDA events",
    "cuda_event_timings_ms": timings_ms,
}
print(json.dumps(result, indent=2))
