import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=1, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=1, suite="stage-b-test-1-gpu-small-amd")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "q_len,kv_len,window,query_offset",
    [
        (4, 8, 3, 4),
        (3, 5, 4, 1),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sliding_window_mask_matches_explicit_reference(
    q_len, kv_len, window, query_offset, causal, dtype
):
    device = torch.device("cuda")
    torch.manual_seed(35003)
    heads, head_dim = 2, 16

    mask = TorchNativeAttnBackend._make_sliding_window_mask(
        q_len=q_len,
        kv_len=kv_len,
        sliding_window_size=window,
        device=device,
        query_offset=query_offset,
        causal=causal,
    )

    q_pos = torch.arange(query_offset, query_offset + q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    if causal:
        expected_mask = (k_pos <= q_pos) & (k_pos >= q_pos - window)
    else:
        expected_mask = torch.abs(q_pos - k_pos) <= window

    assert torch.equal(mask, expected_mask)

    q = torch.randn(1, heads, q_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, heads, kv_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, heads, kv_len, head_dim, device=device, dtype=dtype)

    output = scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask[None, None, :, :],
        is_causal=False,
    )

    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float())
    scores = scores / math.sqrt(head_dim)
    scores = scores.masked_fill(~expected_mask[None, None, :, :], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    reference = torch.einsum("bhqk,bhkd->bhqd", weights, v.float()).to(dtype)

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(
        output.float(),
        reference.float(),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize(
    "attn_type,expected_causal",
    [
        (AttentionType.DECODER, True),
        (AttentionType.ENCODER_ONLY, False),
    ],
)
def test_forward_extend_preserves_sliding_window_and_causality(
    attn_type, expected_causal, monkeypatch
):
    backend = object.__new__(TorchNativeAttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _: torch.empty(1, 1, 8),
        get_value_buffer=lambda _: torch.empty(1, 1, 8),
    )
    backend.req_to_token_pool = SimpleNamespace(req_to_token=torch.empty(1, 1))
    backend.swa_out_cache_loc = None

    captured = {}

    def fake_run_sdpa_forward_extend(*args, **kwargs):
        captured.update(kwargs)
        return None

    backend._run_sdpa_forward_extend = fake_run_sdpa_forward_extend

    layer = SimpleNamespace(
        qk_head_dim=8,
        v_head_dim=8,
        tp_q_head_num=1,
        tp_k_head_num=1,
        is_cross_attention=False,
        attn_type=attn_type,
        sliding_window_size=3,
        scaling=1.0,
        layer_id=0,
    )
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        extend_prefix_lens=torch.tensor([0]),
        extend_seq_lens=torch.tensor([1]),
        encoder_lens=None,
        out_cache_loc=torch.tensor([0]),
    )
    q = torch.zeros(1, 1, 8)
    k = torch.zeros(1, 1, 8)
    v = torch.zeros(1, 1, 8)

    backend.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=False)

    assert captured["causal"] is expected_causal
    assert captured["sliding_window_size"] == 3
