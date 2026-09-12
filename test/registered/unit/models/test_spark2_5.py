"""Tests for Spark2.5 attention output-gate activation selection."""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.configs.spark2_5 import Spark2_5Config
from sglang.srt.models.spark2_5 import _apply_gate_activation
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_config_preserves_gate_activation_mode():
    assert Spark2_5Config().gate_attn_act_mode == "sigmoid"
    assert Spark2_5Config(gate_attn_act_mode="silu").gate_attn_act_mode == "silu"


@pytest.mark.parametrize(
    ("mode", "reference"),
    [("sigmoid", torch.sigmoid), ("silu", F.silu)],
)
def test_gate_activation_matches_float32_reference(mode, reference):
    # Negative and large positive values distinguish SiLU from sigmoid, while
    # zero and ordinary values exercise shared boundary cases.
    gate = torch.tensor([-20.0, -2.0, 0.0, 2.0, 20.0], dtype=torch.float16)

    actual = _apply_gate_activation(gate, mode)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference(gate.float()), rtol=0, atol=0)


def test_gate_activation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported gate_attn_act_mode: gelu"):
        _apply_gate_activation(torch.ones(1), "gelu")
