"""CPU tests for the Kimi-K3 route/quant fusion handoff."""

from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import route_quant_handoff


def _try_fused():
    scores = torch.empty((1, 896), dtype=torch.bfloat16)
    bias = torch.empty(896, dtype=torch.float32)
    x = torch.empty((1, 3584), dtype=torch.bfloat16)
    route_quant_handoff.stage(x)
    return route_quant_handoff.try_route_quant_fused(
        scores,
        bias,
        16,
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )


def test_route_quant_fusion_enabled_by_default():
    envs.SGLANG_DISABLE_KIMI_K3_ROUTE_QUANT_FUSION.set(None)
    outputs = tuple(object() for _ in range(5))
    with (
        patch("sglang.kernels.ops.moe.moe_route_quant_fused.covered", return_value=True),
        patch("sglang.kernels.ops.moe.moe_route_quant_fused.available", return_value=True),
        patch(
            "sglang.kernels.ops.moe.moe_route_quant_fused.route_quant_fused",
            return_value=outputs,
        ) as fused,
    ):
        assert _try_fused() == outputs[:2]
        fused.assert_called_once()
    route_quant_handoff.clear()


def test_route_quant_fusion_can_be_disabled_without_probing_jit():
    envs.SGLANG_DISABLE_KIMI_K3_ROUTE_QUANT_FUSION.set(True)
    with (
        patch("sglang.kernels.ops.moe.moe_route_quant_fused.covered") as covered,
        patch("sglang.kernels.ops.moe.moe_route_quant_fused.available") as available,
        patch("sglang.kernels.ops.moe.moe_route_quant_fused.route_quant_fused") as fused,
    ):
        assert _try_fused() is None
        covered.assert_not_called()
        available.assert_not_called()
        fused.assert_not_called()
    route_quant_handoff.clear()
    envs.SGLANG_DISABLE_KIMI_K3_ROUTE_QUANT_FUSION.set(None)


def test_route_quant_fusion_can_be_reenabled_after_disable():
    envs.SGLANG_DISABLE_KIMI_K3_ROUTE_QUANT_FUSION.set(False)
    outputs = tuple(object() for _ in range(5))
    with (
        patch("sglang.kernels.ops.moe.moe_route_quant_fused.covered", return_value=True),
        patch("sglang.kernels.ops.moe.moe_route_quant_fused.available", return_value=True),
        patch(
            "sglang.kernels.ops.moe.moe_route_quant_fused.route_quant_fused",
            return_value=outputs,
        ) as fused,
    ):
        assert _try_fused() == outputs[:2]
        fused.assert_called_once()
    route_quant_handoff.clear()
    envs.SGLANG_DISABLE_KIMI_K3_ROUTE_QUANT_FUSION.set(None)
