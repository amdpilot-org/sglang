import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="base-a-test-cpu")
npu_moe_methods = importlib.import_module(
    "sglang.srt.hardware_backend.npu.quantization.moe_methods"
)


def _reload_separate_expert_weights(layer, gate, up, down):
    loader = SimpleNamespace(
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_method=SimpleNamespace(load_up_proj_weight_first=False),
        quant_config=None,
        moe_tp_size=1,
        use_padded_loading=False,
        use_presharded_weights=False,
        use_triton_kernels=False,
    )
    FusedMoE._load_w13(loader, layer.w13_weight[0], 0, "w1", gate, 0)
    FusedMoE._load_w13(loader, layer.w13_weight[0], 0, "w3", up, 0)
    FusedMoE._load_w2(loader, layer.w2_weight[0], 1, "w2", down, 0)


@pytest.mark.parametrize("intermediate_size", [1, 3])
def test_npu_postprocess_preserves_layout_for_separate_expert_reload(
    intermediate_size,
):
    """NPU post-processing must not make a second disk load use kernel layout."""
    num_experts, hidden_size = 1, 8
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size, hidden_size),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, intermediate_size),
            requires_grad=False,
        ),
        w13_kernel=npu_moe_methods.NPUUnquantMoEMethod(),
        w2_kernel=npu_moe_methods.NPUUnquantMoEMethod(),
        dispatcher=MagicMock(),
    )
    backend = SimpleNamespace(is_flashinfer_cutlass=lambda: False)

    with (
        patch.object(unquant, "_is_cpu", False),
        patch.object(unquant, "_is_npu", True),
        patch.object(unquant, "_use_aiter", False),
        patch.object(unquant, "get_moe_runner_backend", return_value=backend),
        patch.object(
            npu_moe_methods,
            "npu_format_cast",
            side_effect=lambda tensor: tensor.detach().clone(),
        ),
    ):
        unquant.UnquantizedFusedMoEMethod().process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (num_experts, 2 * intermediate_size, hidden_size)
    assert layer.w2_weight.shape == (num_experts, hidden_size, intermediate_size)

    gate = torch.full((intermediate_size, hidden_size), 1.0)
    up = torch.full((intermediate_size, hidden_size), 2.0)
    down = torch.full((hidden_size, intermediate_size), 3.0)
    _reload_separate_expert_weights(layer, gate, up, down)

    torch.testing.assert_close(layer.w13_weight[0, :intermediate_size], gate)
    torch.testing.assert_close(layer.w13_weight[0, intermediate_size:], up)
    torch.testing.assert_close(layer.w2_weight[0], down)


def test_separate_expert_reload_overwrites_both_halves_independently():
    """A reload must not retain either half of the initially loaded w13 tensor."""
    intermediate_size, hidden_size = 3, 8
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.full((1, 2 * intermediate_size, hidden_size), -1.0),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.full((1, hidden_size, intermediate_size), -1.0),
            requires_grad=False,
        ),
    )

    gate = torch.arange(intermediate_size * hidden_size, dtype=torch.float32).view(
        intermediate_size, hidden_size
    )
    up = gate + 100
    down = torch.arange(hidden_size * intermediate_size, dtype=torch.float32).view(
        hidden_size, intermediate_size
    )
    _reload_separate_expert_weights(layer, gate, up, down)

    torch.testing.assert_close(layer.w13_weight[0, :intermediate_size], gate)
    torch.testing.assert_close(layer.w13_weight[0, intermediate_size:], up)
    torch.testing.assert_close(layer.w2_weight[0], down)
