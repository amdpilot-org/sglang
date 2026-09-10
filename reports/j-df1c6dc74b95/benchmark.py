import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
from sglang.srt.utils import get_device_core_count


CASES = [
    (16384, 1),
    (8192, 2),
    (4096, 4),
    (2048, 8),
    (1024, 16),
    (512, 32),
    (256, 64),
    (128, 128),
]


def reference(q, k, v, kv_indices, sm_scale):
    keys = k.index_select(0, kv_indices).float()
    values = v.index_select(0, kv_indices).float()
    logits = torch.einsum("hd,lhd->hl", q[0].float(), keys) * sm_scale
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probabilities = torch.softmax(logits, dim=-1)
    return torch.einsum("hl,lhd->hd", probabilities, values)


def make_adversarial_inputs(seq_len, heads, head_dim, device, dtype):
    q = torch.where(
        torch.arange(heads, device=device).view(1, heads, 1) % 2 == 0,
        torch.full((1, heads, head_dim), -7.0, device=device),
        torch.full((1, heads, head_dim), 7.0, device=device),
    ).to(dtype)
    k = torch.where(
        torch.arange(seq_len, device=device).view(-1, 1, 1) % 2 == 0,
        torch.full((seq_len, heads, head_dim), -6.0, device=device),
        torch.full((seq_len, heads, head_dim), 6.0, device=device),
    ).to(dtype)
    v = torch.where(
        torch.arange(seq_len, device=device).view(-1, 1, 1) % 3 == 0,
        torch.full((seq_len, heads, head_dim), -2.0, device=device),
        torch.full((seq_len, heads, head_dim), 2.0, device=device),
    ).to(dtype)
    return q, k, v


def run_case(seq_len, heads, head_dim, max_kv_splits, core_count, device, dtype):
    q, k, v = make_adversarial_inputs(seq_len, heads, head_dim, device, dtype)
    output = torch.zeros_like(q)
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(seq_len, device=device)
    seq_lens = torch.full((1,), seq_len, dtype=torch.int32, device=device)
    num_kv_splits = torch.empty((1,), dtype=torch.int32, device=device)
    get_num_kv_splits_triton[(1,)](
        num_kv_splits,
        seq_lens,
        1,
        1,
        heads,
        heads,
        max_kv_splits,
        core_count,
        MAX_NUM_SEQ=256,
    )
    attn_logits = torch.empty(
        (1, heads, max_kv_splits, head_dim), dtype=torch.float32, device=device
    )
    attn_lse = torch.empty(
        (1, heads, max_kv_splits), dtype=torch.float32, device=device
    )
    sm_scale = head_dim ** -0.5
    args = (
        q,
        k,
        v,
        output,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        1.0,
        1.0,
    )
    decode_attention_fwd(*args)
    torch.cuda.synchronize()
    reference_output = reference(q, k, v, kv_indices, sm_scale)
    max_abs_error = (output.float() - reference_output).abs().max().item()
    passed = torch.allclose(
        output.float(), reference_output, atol=1e-2, rtol=1e-2
    )

    for _ in range(3):
        decode_attention_fwd(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        decode_attention_fwd(*args)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    flops = 4.0 * seq_len * heads * head_dim
    bytes_moved = 2.0 * seq_len * heads * head_dim + 2.0 * heads * head_dim
    return {
        "seq_len": seq_len,
        "heads": heads,
        "num_kv_splits": int(num_kv_splits.item()),
        "numerical_gate_passed": bool(passed),
        "max_abs_error": max_abs_error,
        "median_ms": sorted(samples)[len(samples) // 2],
        "samples_ms": samples,
        "estimated_arithmetic_intensity": flops / bytes_moved,
    }


def main():
    device = "cuda"
    dtype = torch.bfloat16
    head_dim = 64
    max_kv_splits = int(os.environ.get("MAX_KV_SPLITS", "16"))
    core_count = get_device_core_count(0)
    results = [
        run_case(
            seq_len,
            heads,
            head_dim,
            max_kv_splits,
            core_count,
            device,
            dtype,
        )
        for seq_len, heads in CASES
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
