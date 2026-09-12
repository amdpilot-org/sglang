from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.quantization import fp8 as fp8_quant


def _method(*, is_fp4_expert: bool, dequant: bool):
    method = fp8_quant.Fp8MoEMethod.__new__(fp8_quant.Fp8MoEMethod)
    method.convert_mxfp8_to_block = False
    method.use_mxfp8 = False
    method.is_fp4_expert = is_fp4_expert
    method.dequant_fp4_to_fp8 = dequant
    method.quant_config = type("Config", (), {"is_checkpoint_fp8_serialized": True})()
    return method


def _layer():
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros((1, 128, 64), dtype=torch.int8), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros((1, 128, 64), dtype=torch.int8), requires_grad=False
    )
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones((1, 128, 4)), requires_grad=False
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.ones((1, 128, 4)), requires_grad=False
    )
    return layer


def _fake_cast(weight, scale):
    return (
        torch.zeros((weight.shape[0], weight.shape[1] * 2), dtype=torch.float8_e4m3fn),
        torch.ones((weight.shape[0] // 128, weight.shape[1] * 2 // 128)),
    )


@pytest.mark.parametrize("fp8_fnuz", [False, True])
def test_fp4_dequant_precedes_rocm_weight_processing(fp8_fnuz):
    method = _method(is_fp4_expert=True, dequant=True)
    layer = _layer()

    with (
        patch.multiple(
            fp8_quant,
            _use_aiter=True,
            _is_fp8_fnuz=fp8_fnuz,
            _is_cpu=False,
        ),
        patch.object(fp8_quant, "cast_e2m1fn_to_e4m3fn", side_effect=_fake_cast) as cast,
        patch.object(
            fp8_quant,
            "normalize_e4m3fn_to_e4m3fnuz",
            side_effect=lambda weight, weight_scale, input_scale: (
                weight,
                weight_scale,
                input_scale,
            ),
        ),
        patch.object(
            fp8_quant,
            "shuffle_weight",
            side_effect=lambda weight, *args: weight,
            create=True,
        ),
    ):
        method.process_weights_after_loading_block_quant(layer)

    assert cast.call_count == 2
    assert method.is_fp4_expert is False
    assert layer.w13_weight.shape == (1, 128, 128)
    assert layer.w2_weight.shape == (1, 128, 128)


@pytest.mark.parametrize(
    ("is_fp4_expert", "dequant"), [(True, False), (False, True)]
)
def test_fp4_dequant_requires_both_fp4_weights_and_flag(is_fp4_expert, dequant):
    method = _method(is_fp4_expert=is_fp4_expert, dequant=dequant)
    method.convert_mxfp8_to_block = True
    method._convert_mxfp8_moe_to_block_fp8 = lambda layer: None
    layer = _layer()

    with (
        patch.multiple(fp8_quant, _use_aiter=False, _is_fp8_fnuz=False, _is_cpu=False),
        patch.object(fp8_quant, "cast_e2m1fn_to_e4m3fn") as cast,
    ):
        method.process_weights_after_loading_block_quant(layer)

    cast.assert_not_called()
