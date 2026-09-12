"""Independent gfx950 numerical check for QSA chunk-prefill FP8 cache reads."""

import json

import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
)


def reference(q, k, v, scale):
    outputs = []
    for row in range(q.shape[0]):
        scores = torch.einsum(
            "hd,kd->hk", q[row].float(), k[row].float()
        ) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,kd->hd", probs, v[row].float()))
    return torch.stack(outputs).to(q.dtype)


def run_case(topk, cache_dtype):
    requests, q_heads, kv_heads, head_dim = 2, 4, 1, 128
    q = torch.randn(requests, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(requests, topk, head_dim, device="cuda", dtype=torch.bfloat16)
    v_bf16 = torch.randn(requests, topk, head_dim, device="cuda", dtype=torch.bfloat16)
    k = k_bf16.reshape(requests * topk, kv_heads, head_dim).to(cache_dtype)
    v = v_bf16.reshape(requests * topk, kv_heads, head_dim).to(cache_dtype)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(requests, 1)
    cu_q = torch.arange(requests + 1, device="cuda", dtype=torch.int32)
    cu_k = torch.arange(0, (requests + 1) * topk, topk, device="cuda", dtype=torch.int32)
    kv_lens = torch.full((requests,), topk, device="cuda", dtype=torch.int32)
    scale = head_dim**-0.5
    actual = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_q, cu_k, kv_lens, scale
    )
    expected = reference(
        q,
        k.reshape(requests, topk, head_dim).to(torch.bfloat16),
        v.reshape(requests, topk, head_dim).to(torch.bfloat16),
        scale,
    )
    error = (actual - expected).abs().float()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    return {"topk": topk, "cache_dtype": str(cache_dtype), "max_abs_error": error.max().item()}


torch.manual_seed(36545)
properties = torch.cuda.get_device_properties(0)
results = []
for dtype in (torch.bfloat16, torch.float8_e4m3fn):
    for topk in (1, 16, 17):
        results.append(run_case(topk, dtype))
print(json.dumps({"device": torch.cuda.get_device_name(0), "arch": properties.gcnArchName, "cases": results}, indent=2))
