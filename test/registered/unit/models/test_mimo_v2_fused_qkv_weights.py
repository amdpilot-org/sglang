from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.models.mimo_v2 import (
    _resolve_deferred_qkv_scale_inv,
    get_mimo_v2_fused_qkv_expected_tp_size,
    load_mimo_v2_qkv_proj_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _parallel(tp_size, tp_rank):
    return SimpleNamespace(attn_tp_size=tp_size, attn_tp_rank=tp_rank)


def test_fused_qkv_checkpoint_merges_interleaved_shards_for_smaller_runtime_tp():
    # A Pro checkpoint stores one fused [Q, K, V] tensor per KV-head shard.
    # At runtime TP=2, each rank must receive four adjacent checkpoint shards,
    # not one eighth of the flattened tensor.
    checkpoint = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    param = torch.nn.Parameter(torch.empty(4, 3), requires_grad=False)

    with patch("sglang.srt.models.mimo_v2.get_parallel", return_value=_parallel(2, 1)):
        load_mimo_v2_qkv_proj_weight(
            "model.layers.0.self_attn.qkv_proj.weight",
            param,
            checkpoint,
            expected_fused_tp_size=8,
        )

    torch.testing.assert_close(param, checkpoint[4:])


def test_fused_qkv_checkpoint_rejects_incompatible_runtime_tp():
    param = torch.nn.Parameter(torch.empty(4, 3), requires_grad=False)
    checkpoint = torch.empty(8, 3)

    with (
        patch("sglang.srt.models.mimo_v2.get_parallel", return_value=_parallel(3, 0)),
        pytest.raises(ValueError, match="TP=8-interleaved"),
    ):
        load_mimo_v2_qkv_proj_weight(
            "model.layers.0.self_attn.qkv_proj.weight",
            param,
            checkpoint,
            expected_fused_tp_size=8,
        )


def test_fused_qkv_scale_requantization_preserves_qkv_order():
    # Two checkpoint shards, each laid out [Q0, Q1, K, V].  Merging them for
    # runtime TP=1 must produce [all Q, all K, all V].  Distinct values make an
    # accidental plain concatenation observable after dequantize/requantize.
    checkpoint_weight = torch.tensor(
        [[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float8_e4m3fn
    )
    checkpoint_scale = torch.ones(4, 1, dtype=torch.float32)
    weight = torch.nn.Parameter(checkpoint_weight.clone(), requires_grad=False)
    scale = torch.nn.Parameter(torch.empty(4, 1), requires_grad=False)
    name = "model.layers.0.self_attn.qkv_proj.weight_scale_inv"
    config = SimpleNamespace(
        hybrid_layer_pattern=[0],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=1,
        v_head_dim=1,
    )

    with patch("sglang.srt.models.mimo_v2.get_parallel", return_value=_parallel(1, 0)):
        _resolve_deferred_qkv_scale_inv(
            {
                name.replace(".weight_scale_inv", ".weight"): weight,
                name: scale,
            },
            {name: checkpoint_scale},
            expected_fused_tp_size=2,
            block_size=2,
            config=config,
        )

    recovered = (
        weight.float() * scale.repeat_interleave(2, 0).repeat_interleave(2, 1)[:8, :1]
    )
    expected = torch.tensor(
        [[1], [2], [5], [6], [3], [7], [4], [8]], dtype=torch.float32
    )
    legacy_plain_concatenation = checkpoint_weight.float()
    assert not torch.equal(legacy_plain_concatenation, expected)
    # FP8 requantization introduces the expected small rounding error.
    torch.testing.assert_close(recovered, expected, rtol=0.03, atol=0.02)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(attention_projection_layout=None), None),
        (
            SimpleNamespace(
                attention_projection_layout="fused_qkv", num_key_value_heads=32
            ),
            32,
        ),
    ],
)
def test_fused_qkv_expected_tp_size(config, expected):
    assert get_mimo_v2_fused_qkv_expected_tp_size(config) == expected


def test_fused_qkv_expected_tp_size_rejects_unknown_layout():
    with pytest.raises(ValueError, match="unsupported"):
        get_mimo_v2_fused_qkv_expected_tp_size(
            SimpleNamespace(attention_projection_layout="split_qkv")
        )
