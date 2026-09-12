"""CPU regressions for VisionAttention's external FA4 call boundary."""

import pytest
import torch
from torch import nn

from sglang.srt.layers.attention import vision
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _reference_attention(q, k, v, window_size, sinks, softmax_scale):
    """Small independent reference for equal-length, non-causal FA attention."""
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * softmax_scale
    left, right = window_size
    if left >= 0 or right >= 0:
        positions = torch.arange(q.shape[0])
        distance = positions[:, None] - positions[None, :]
        allowed = torch.ones_like(distance, dtype=torch.bool)
        if left >= 0:
            allowed &= distance <= left
        if right >= 0:
            allowed &= distance >= -right
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))

    maximum = scores.amax(dim=-1, keepdim=True)
    if sinks is not None:
        maximum = torch.maximum(maximum, sinks.float()[:, None, None])
    weights = torch.exp(scores - maximum)
    denominator = weights.sum(dim=-1, keepdim=True)
    if sinks is not None:
        denominator += torch.exp(sinks.float()[:, None, None] - maximum)
    return torch.einsum("hqk,khd->qhd", weights / denominator, v.float()).to(q)


@pytest.mark.parametrize(
    ("forward_kwargs", "expected_window", "expect_sinks"),
    [
        ({}, (-1, -1), False),
        ({"window_size": (2, 1)}, (2, 1), False),
        ({"s_aux": torch.tensor([0.25])}, (-1, -1), True),
        (
            {"window_size": (2, 1), "s_aux": torch.tensor([0.25])},
            (2, 1),
            True,
        ),
    ],
)
def test_vision_flash4_forwards_optional_attention_semantics(
    monkeypatch, forward_kwargs, expected_window, expect_sinks
):
    captured = {}

    def external_fa4_boundary(q, k, v, **kwargs):
        captured.update(kwargs)
        return q

    monkeypatch.setattr(vision, "_is_cuda", True)
    monkeypatch.setattr(
        vision, "flash_attn_func", external_fa4_boundary, raising=False
    )
    attention = vision.VisionFlash4Attention()
    q = torch.zeros(4, 1, 8)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

    assert (
        attention(q, q, q, cu_seqlens, bsz=1, seq_len=4, **forward_kwargs) is q
    )
    assert captured["window_size"] == expected_window
    assert captured["ver"] == 4
    if expect_sinks:
        assert captured["sinks"] is forward_kwargs["s_aux"]
    else:
        assert "sinks" not in captured


class _FixedQKV(nn.Module):
    def __init__(self, q, k, v):
        super().__init__()
        self.qkv = torch.cat((q, k, v), dim=-1).unsqueeze(0)

    def forward(self, _):
        return self.qkv, None


class _IdentityProjection(nn.Module):
    def forward(self, x):
        return x, None


def test_vision_attention_preserves_local_window_and_sink_at_fa4_boundary(
    monkeypatch,
):
    """Exercise VisionAttention -> VisionFlash4Attention, including semantics."""
    q = torch.tensor([[[-1.0, 0.5]], [[0.2, 1.1]], [[0.7, -0.4]], [[1.2, 0.3]]])
    k = torch.tensor([[[0.4, -0.8]], [[1.0, 0.2]], [[-0.3, 0.9]], [[0.6, 0.7]]])
    v = torch.tensor([[[1.0, 2.0]], [[3.0, -1.0]], [[-2.0, 0.5]], [[0.4, 4.0]]])
    window_size = (1, 0)
    sinks = torch.tensor([0.6])
    scale = 0.5
    captured = {}

    def external_fa4_boundary(q_arg, k_arg, v_arg, **kwargs):
        captured.update(kwargs)
        return _reference_attention(
            q_arg,
            k_arg,
            v_arg,
            kwargs.get("window_size", (-1, -1)),
            kwargs.get("sinks"),
            kwargs["softmax_scale"],
        )

    monkeypatch.setattr(
        vision, "flash_attn_func", external_fa4_boundary, raising=False
    )

    # Build only the ordinary VisionAttention data path; fixed projections keep
    # this regression independent of distributed initialization and GPU kernels.
    layer = vision.VisionAttention.__new__(vision.VisionAttention)
    nn.Module.__init__(layer)
    layer.qkv_proj = _FixedQKV(q, k, v)
    layer.proj = _IdentityProjection()
    layer.qkv_backend = vision.VisionFlash4Attention.__new__(
        vision.VisionFlash4Attention
    )
    nn.Module.__init__(layer.qkv_backend)
    layer.use_qkv_parallel = True
    layer.pass_strided_qkv = False
    layer.q_size = layer.kv_size = layer.head_size = 2
    layer.num_attention_heads_per_partition = 1
    layer.num_attention_kv_heads_per_partition = 1
    layer.qk_normalization = False
    layer.qk_normalization_by_head_size = False
    layer.customized_position_embedding_applier = None
    layer.softmax_scale = scale
    layer.sinks = nn.Parameter(sinks, requires_grad=False)
    layer.window_size = window_size
    layer.tp_rank = 0

    with get_context().override_server_args():
        output = layer(
            torch.empty(1, 4, 2),
            cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            max_seqlen=4,
            full_attn=False,
        )
    expected = _reference_attention(q, k, v, window_size, sinks, scale).reshape(
        1, 4, 2
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert captured["window_size"] == window_size
    assert torch.equal(captured["sinks"], layer.sinks)
    assert captured["sinks"].untyped_storage().data_ptr() == (
        layer.sinks.untyped_storage().data_ptr()
    )
    assert captured["ver"] == 4
