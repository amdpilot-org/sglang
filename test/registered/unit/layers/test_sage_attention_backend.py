from unittest.mock import patch

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.sage_attention_backend import SageAttentionBackend
from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _backend_with_reference_kernel():
    backend = SageAttentionBackend.__new__(SageAttentionBackend)
    calls = []

    def reference(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return scaled_dot_product_attention(
            q,
            k,
            v,
            scale=kwargs["sm_scale"],
            is_causal=kwargs["is_causal"],
        )

    backend.sageattn = reference
    return backend, calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sage_adapter_matches_independent_sdpa_reference_for_gqa(dtype):
    torch.manual_seed(7)
    device = torch.device("cuda")
    q = torch.randn(1, 4, 17, 64, device=device, dtype=dtype)
    k = torch.randn(1, 2, 17, 64, device=device, dtype=dtype)
    v = torch.randn(1, 2, 17, 64, device=device, dtype=dtype)
    scale = 0.125
    backend, calls = _backend_with_reference_kernel()

    actual = backend._attention(
        q,
        k,
        v,
        attn_mask=None,
        enable_gqa=True,
        scale=scale,
        is_causal=True,
    )
    expected = scaled_dot_product_attention(
        q, k, v, enable_gqa=True, scale=scale, is_causal=True
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    sage_q, sage_k, sage_v, kwargs = calls[0]
    assert sage_q.shape == sage_k.shape == sage_v.shape == (1, 4, 17, 64)
    assert kwargs == {
        "tensor_layout": "HND",
        "is_causal": True,
        "sm_scale": scale,
    }


@pytest.mark.parametrize(
    ("shape", "dtype", "mask", "message"),
    [
        ((1, 2, 4, 32), torch.float16, None, "head dimensions"),
        ((1, 2, 4, 64), torch.float32, None, "float16 or bfloat16"),
        ((1, 2, 4, 64), torch.float16, torch.ones(4, 4), "masks"),
    ],
)
def test_sage_adapter_rejects_unsupported_inputs(shape, dtype, mask, message):
    backend, _ = _backend_with_reference_kernel()
    q = torch.empty(shape, dtype=dtype)
    with pytest.raises(ValueError, match=message):
        backend._attention(
            q,
            q,
            q,
            attn_mask=mask,
            enable_gqa=False,
            scale=None,
            is_causal=False,
        )


def test_sage_adapter_rejects_invalid_gqa_ratio():
    backend, _ = _backend_with_reference_kernel()
    q = torch.empty(1, 3, 4, 64, dtype=torch.float16)
    kv = torch.empty(1, 2, 4, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="divisible"):
        backend._attention(
            q,
            kv,
            kv,
            attn_mask=None,
            enable_gqa=True,
            scale=None,
            is_causal=False,
        )


def test_sage_backend_fails_closed_on_rocm():
    with (
        patch(
            "sglang.srt.layers.attention.sage_attention_backend.is_hip",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="requires NVIDIA CUDA"),
    ):
        SageAttentionBackend(object())
