import json
import os
import subprocess
import time
from pathlib import Path

import torch

from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton,
)

START = time.perf_counter()
REPO = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda:0")


def make_indices(sequence_lengths, topk):
    total = sum(sequence_lengths)
    indices = torch.full((total, topk), -1, dtype=torch.int32, device=DEVICE)
    slots = torch.full_like(indices, -1)
    for batch, sequence_length in enumerate(sequence_lengths):
        start = sum(sequence_lengths[:batch])
        for position in range(sequence_length):
            valid_count = min(topk, position + 1)
            selected = torch.randperm(position + 1, device=DEVICE)[:valid_count]
            indices[start + position, :valid_count] = selected.to(torch.int32)
            slots[start + position, :valid_count] = start + selected.to(torch.int32)
    return indices, slots


def make_tensor(rows, heads, head_dim, non_contiguous, sentinel):
    if not non_contiguous:
        return (
            torch.randn(rows, heads, head_dim, device=DEVICE, dtype=torch.bfloat16),
            None,
            None,
            None,
        )
    storage = torch.full(
        (rows * heads * head_dim * 2 + 32,),
        sentinel,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    view = storage[16 : 16 + rows * heads * head_dim * 2].view(
        rows, heads, head_dim * 2
    )
    tensor = view[:, :, :head_dim]
    tensor.normal_()
    return tensor, storage, storage[:16].clone(), storage[-16:].clone()


def run_case(
    sequence_lengths, num_q_heads, num_kv_heads, head_dim, topk, non_contiguous
):
    total = sum(sequence_lengths)
    q, q_storage, q_before, q_after = make_tensor(
        total, num_q_heads, head_dim, non_contiguous, -123.0
    )
    k, k_storage, k_before, k_after = make_tensor(
        total, num_kv_heads, head_dim, non_contiguous, 321.0
    )
    v, v_storage, v_before, v_after = make_tensor(
        total, num_kv_heads, head_dim, non_contiguous, 654.0
    )
    indices, slots = make_indices(sequence_lengths, topk)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(sequence_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    scale = head_dim**-0.5
    actual = sparse_gqa_fwd_interface_triton(
        q,
        k,
        v,
        max(sequence_lengths),
        indices,
        cu_seqlens,
        scale,
    )
    torch.cuda.synchronize()
    expected = qsa_sparse_attention(q, k, v, slots, scale)
    difference = (actual.float() - expected.float()).abs()

    for _ in range(5):
        sparse_gqa_fwd_interface_triton(
            q, k, v, max(sequence_lengths), indices, cu_seqlens, scale
        )
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(20):
        sparse_gqa_fwd_interface_triton(
            q, k, v, max(sequence_lengths), indices, cu_seqlens, scale
        )
    end_event.record()
    torch.cuda.synchronize()

    guards_unchanged = True
    if non_contiguous:
        guards_unchanged = all(
            (
                torch.equal(q_storage[:16], q_before),
                torch.equal(q_storage[-16:], q_after),
                torch.equal(k_storage[:16], k_before),
                torch.equal(k_storage[-16:], k_after),
                torch.equal(v_storage[:16], v_before),
                torch.equal(v_storage[-16:], v_after),
            )
        )
    return {
        "sequence_lengths": sequence_lengths,
        "query_heads": num_q_heads,
        "kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "topk": topk,
        "representation": "non-contiguous" if non_contiguous else "contiguous",
        "q_stride": list(q.stride()),
        "k_stride": list(k.stride()),
        "v_stride": list(v.stride()),
        "iterations": 20,
        "mean_ms": start_event.elapsed_time(end_event) / 20.0,
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "guards_unchanged": guards_unchanged,
    }


torch.manual_seed(247)
cases = [
    ([17, 31], 4, 2, 64, 8),
    ([64, 128], 4, 2, 64, 8),
    ([64, 128], 8, 2, 128, 8),
]
results = []
first_execution_elapsed = None
for case in cases:
    for non_contiguous in (False, True):
        result = run_case(*case, non_contiguous)
        if first_execution_elapsed is None:
            first_execution_elapsed = time.perf_counter() - START
        results.append(result)

cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", "~/.triton")).expanduser()
native_artifacts = sorted(
    str(path) for pattern in ("*.hsaco", "*.cubin") for path in cache_dir.rglob(pattern)
)[:8]
try:
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        _resolve_flash_attn_varlen_func,
    )

    qsa_varlen_fallback = {
        "available": True,
        "resolved": str(_resolve_flash_attn_varlen_func()),
    }
except Exception as error:
    qsa_varlen_fallback = {
        "available": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }
report = {
    "label": "mirror-checkout gfx942 execution-representation result",
    "base_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip(),
    "working_tree_dirty": True,
    "gpu": {
        "name": torch.cuda.get_device_name(DEVICE),
        "capability": torch.cuda.get_device_capability(DEVICE),
        "torch": torch.__version__,
        "torch_path": torch.__file__,
        "rocm": torch.version.hip,
    },
    "image": {
        "required": "amdpilotv2/open-job-mi300:jit-config-readable-35122-260909",
        "local_id": (
            "sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1"
        ),
        "identity_source": "operator-provided local image ID",
    },
    "operation": "sparse_gqa_fwd_interface_triton unpacked Q/K/V prefill",
    "source_paths": {
        "python_wrapper": str(
            REPO / "python/sglang/srt/layers/attention/qsa/sparse_attn.py"
        ),
        "test": str(REPO / "test/registered/kernel/qsa/test_qsa.py"),
    },
    "native_dispatch": {
        "kernel": "sglang.srt.layers.attention.qsa.sparse_attn._sparse_gqa_prefill",
        "compiler": "Triton",
        "target": "gfx942",
        "cache_dir": str(cache_dir),
        "native_artifacts": native_artifacts,
    },
    "qsa_varlen_fallback": qsa_varlen_fallback,
    "reference": "qsa_sparse_attention independent float32 PyTorch reference",
    "timing_method": "5 warmups, 20 timed calls, one CUDA event pair per case",
    "first_gpu_execution_elapsed_seconds": first_execution_elapsed,
    "results": results,
}
output = Path(__file__).with_name("results.json")
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
