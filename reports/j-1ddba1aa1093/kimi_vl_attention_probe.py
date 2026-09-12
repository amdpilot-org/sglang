"""GPU evidence for the already-landed Kimi-VL packed-attention fix.

This is deliberately a kernel-level fixture. It validates packed attention
boundaries, numerical output, and transient allocation behavior, but it is not
a Kimi-VL/MMMU serving reproduction.
"""

import json
import math

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.vision import (
    VisionTritonAttention,
    prepare_vision_attention_metadata,
)
from sglang.srt.runtime_context import get_parallel


def reference(q, k, v, boundaries):
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        outputs.append(
            F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
    return torch.cat(outputs)


def legacy_dense_sdpa(q, k, v, boundaries):
    total_tokens = q.shape[0]
    mask = torch.zeros(
        (1, total_tokens, total_tokens), device=q.device, dtype=torch.bool
    )
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mask[..., start:end, start:end] = True
    output = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=mask,
        dropout_p=0.0,
    )
    return output.transpose(0, 1), mask.numel() * mask.element_size()


def measured_call(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    value = fn()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    return value, peak - max(before, after)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    torch.manual_seed(7433)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    heads = 4
    head_dim = 32
    with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0):
        backend = VisionTritonAttention(use_data_parallel=False).to(device)

    cases = []
    for name, lengths in (("single", [17]), ("uneven", [11, 23, 7])):
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        q, k, v = [
            torch.randn(boundaries[-1], heads, head_dim, device=device, dtype=dtype)
            for _ in range(3)
        ]
        cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
        metadata = prepare_vision_attention_metadata(cu, device=device)
        actual = backend(
            q,
            k,
            v,
            cu_seqlens=cu,
            bsz=len(lengths),
            seq_len=max(lengths),
            softmax_scale=1 / math.sqrt(head_dim),
            forward_metadata=metadata,
        )
        expected = reference(q, k, v, boundaries)
        torch.cuda.synchronize()
        cases.append(
            {
                "name": name,
                "lengths": lengths,
                "max_abs_diff": (actual.float() - expected.float()).abs().max().item(),
                "all_finite": bool(torch.isfinite(actual).all().item()),
            }
        )

    lengths = [1024, 1536]
    boundaries = [0, lengths[0], sum(lengths)]
    q, k, v = [
        torch.randn(boundaries[-1], heads, head_dim, device=device, dtype=dtype)
        for _ in range(3)
    ]
    cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
    metadata = prepare_vision_attention_metadata(cu, device=device)

    backend(q, k, v, cu, len(lengths), max(lengths), forward_metadata=metadata)
    reference(q, k, v, boundaries)
    torch.cuda.synchronize()
    current, current_peak = measured_call(
        lambda: backend(
            q,
            k,
            v,
            cu_seqlens=cu,
            bsz=len(lengths),
            seq_len=max(lengths),
            forward_metadata=metadata,
        )
    )
    (legacy_output, dense_mask_bytes), legacy_peak = measured_call(
        lambda: legacy_dense_sdpa(q, k, v, boundaries)
    )
    result = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "backend": "triton_attn",
        "boundary_cases": cases,
        "memory_case_lengths": lengths,
        "current_peak_intermediate_bytes": current_peak,
        "legacy_peak_intermediate_bytes": legacy_peak,
        "legacy_dense_mask_bytes": dense_mask_bytes,
        "memory_output_max_abs_diff": (current.float() - legacy_output.float())
        .abs()
        .max()
        .item(),
    }
    print(json.dumps(result, indent=2))
    assert all(case["all_finite"] and case["max_abs_diff"] <= 0.02 for case in cases)
    assert result["memory_output_max_abs_diff"] <= 0.02
    assert legacy_peak > current_peak


if __name__ == "__main__":
    main()
