"""Regression coverage for transformers-written GLM-5.3 checkpoints."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LAYER = "model.language_model.layers.0"


class _LoaderHarness(nn.Module):
    """Small module that exercises the production load_weights mappings."""

    def __init__(self):
        super().__init__()
        self.encoder_only = False
        self.language_only = False
        self.num_fused_shared_experts = 0
        self.fuse_qkv_a_proj = False
        self.quant_config = None
        self.config = SimpleNamespace(n_routed_experts=2, num_hidden_layers=1)
        self.loaded = {}
        self._parameters_by_name = {}

        for name in (
            "model.layers.0.hc_attn_fn",
            "model.layers.0.hc_ffn_scale",
            "model.layers.0.self_attn.fused_qkvbfg_a_proj.weight",
            "model.layers.0.self_attn.fused_fg_b_proj.weight",
            "model.layers.0.self_attn.dt_bias",
            "model.layers.0.self_attn.A_log",
            "model.layers.0.self_attn.qkv_conv1d.weight",
        ):
            self._add_param(name, self._stacked_loader(name))
        for name in (
            "model.layers.0.mlp.experts.w13_weight",
            "model.layers.0.mlp.experts.w2_weight",
        ):
            self._add_param(name, self._expert_loader(name))

    def _add_param(self, name, loader):
        param = nn.Parameter(torch.empty(1))
        param.weight_loader = loader
        self._parameters_by_name[name] = param

    def _stacked_loader(self, name):
        def load(_param, value, shard_id=None):
            self.loaded[(name, shard_id)] = value.clone()

        return load

    def _expert_loader(self, name):
        def load(_param, value, weight_name, shard_id, expert_id):
            self.loaded[(name, expert_id, shard_id)] = value.clone()

        return load

    def named_parameters(self, *args, **kwargs):
        return iter(self._parameters_by_name.items())


def _load(weights):
    model = _LoaderHarness()
    with patch(
        "sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights"
    ):
        Glm5NextForConditionalGeneration.load_weights(model, weights)
    return model.loaded


def test_transformers_layout_exercises_actual_loader_mappings():
    attn_fn = torch.arange(6).reshape(2, 3)
    ffn_scale = torch.arange(3)
    f_a = torch.arange(8).reshape(2, 4)
    f_b = torch.arange(12).reshape(3, 4)
    dt_bias = torch.arange(3)
    a_log = torch.arange(3)
    gate = torch.arange(24).reshape(2, 3, 4)
    up = gate + 100
    down = torch.arange(40).reshape(2, 4, 5)
    conv_parts = [torch.full((2, 1, 3), value) for value in (1, 2, 3)]

    loaded = _load(
        [
            (f"{LAYER}.attn_hc.fn", attn_fn),
            (f"{LAYER}.ffn_hc.scale", ffn_scale),
            (f"{LAYER}.self_attn.forget_gate.f_a_proj.weight", f_a),
            (f"{LAYER}.self_attn.forget_gate.f_b_proj.weight", f_b),
            (f"{LAYER}.self_attn.forget_gate.dt_bias", dt_bias),
            (f"{LAYER}.self_attn.forget_gate.A_log", a_log),
            (f"{LAYER}.mlp.experts.gate_up_proj", torch.cat((gate, up), dim=1)),
            (f"{LAYER}.mlp.experts.down_proj", down),
            (f"{LAYER}.self_attn.conv1d.weight", torch.cat(conv_parts)),
        ]
    )

    torch.testing.assert_close(loaded[("model.layers.0.hc_attn_fn", None)], attn_fn)
    torch.testing.assert_close(loaded[("model.layers.0.hc_ffn_scale", None)], ffn_scale)
    torch.testing.assert_close(
        loaded[("model.layers.0.self_attn.fused_qkvbfg_a_proj.weight", 4)], f_a
    )
    torch.testing.assert_close(
        loaded[("model.layers.0.self_attn.fused_fg_b_proj.weight", 0)], f_b
    )
    torch.testing.assert_close(
        loaded[("model.layers.0.self_attn.dt_bias", None)], dt_bias
    )
    assert loaded[("model.layers.0.self_attn.A_log", None)].shape == (1, 1, 3, 1)
    for expert in range(2):
        torch.testing.assert_close(
            loaded[("model.layers.0.mlp.experts.w13_weight", expert, "w1")],
            gate[expert],
        )
        torch.testing.assert_close(
            loaded[("model.layers.0.mlp.experts.w13_weight", expert, "w3")],
            up[expert],
        )
        torch.testing.assert_close(
            loaded[("model.layers.0.mlp.experts.w2_weight", expert, "w2")],
            down[expert],
        )
    for shard_id, expected in enumerate(conv_parts):
        torch.testing.assert_close(
            loaded[("model.layers.0.self_attn.qkv_conv1d.weight", shard_id)],
            expected,
        )


@pytest.mark.parametrize(
    "name,weight,match",
    [
        (f"{LAYER}.attn_hc.required_new", torch.zeros(1), "Unsupported required"),
        (
            f"{LAYER}.mlp.experts.gate_up_proj",
            torch.zeros(3, 4, 5),
            "expected 2 experts",
        ),
        (
            f"{LAYER}.self_attn.conv1d.weight",
            torch.zeros(6, 2, 3),
            "expected shape.*3P, 1, K",
        ),
    ],
)
def test_unsupported_or_malformed_required_weights_fail(name, weight, match):
    with pytest.raises(ValueError, match=match):
        _load([(name, weight)])


def test_released_layout_still_uses_the_same_loader():
    value = torch.arange(6).reshape(2, 3)
    loaded = _load([(f"{LAYER}.hc_attn_fn", value)])
    torch.testing.assert_close(loaded[("model.layers.0.hc_attn_fn", None)], value)
