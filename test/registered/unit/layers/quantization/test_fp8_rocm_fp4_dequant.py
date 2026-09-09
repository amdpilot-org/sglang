import pytest
import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod


FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


class _FusedMoELayer(torch.nn.Module):
    pass


def _make_layer(num_experts, output_size, input_size, device):
    layer = _FusedMoELayer()
    generator = torch.Generator().manual_seed(1234 + output_size + input_size)
    packed = torch.randint(
        0,
        256,
        (num_experts, output_size, input_size // 2),
        generator=generator,
        dtype=torch.uint8,
    )
    scale_bytes = torch.randint(
        127,
        134,
        (num_experts, output_size, input_size // 32),
        generator=generator,
        dtype=torch.uint8,
    )
    layer.w13_weight = torch.nn.Parameter(
        packed.to(device=device).view(torch.int8), requires_grad=False
    )
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        scale_bytes.to(device=device).view(torch.float8_e8m0fnu), requires_grad=False
    )
    return layer


def _add_w2(layer, num_experts, output_size, input_size, device):
    generator = torch.Generator().manual_seed(4321 + output_size + input_size)
    packed = torch.randint(
        0,
        256,
        (num_experts, output_size, input_size // 2),
        generator=generator,
        dtype=torch.uint8,
    )
    scale_bytes = torch.randint(
        127,
        134,
        (num_experts, output_size, input_size // 32),
        generator=generator,
        dtype=torch.uint8,
    )
    layer.w2_weight = torch.nn.Parameter(
        packed.to(device=device).view(torch.int8), requires_grad=False
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        scale_bytes.to(device=device).view(torch.float8_e8m0fnu), requires_grad=False
    )
    return layer


def _reference_fp4_dequant(packed, scale_bytes):
    packed_uint8 = packed.view(torch.uint8)
    low = FP4_VALUES.to(packed.device)[(packed_uint8 & 0x0F).long()]
    high = FP4_VALUES.to(packed.device)[((packed_uint8 >> 4) & 0x0F).long()]
    values = torch.stack([low, high], dim=-1).flatten(-2)
    scales = torch.exp2(scale_bytes.float() - 127.0)
    return values, scales.repeat_interleave(32, dim=-1)


def _expected_fp8_conversion(packed, scale_bytes):
    values, expanded_scales = _reference_fp4_dequant(packed, scale_bytes)
    output_size, input_size = values.shape[-2:]
    block_values = values.view(
        output_size // 128, 128, input_size // 128, 128
    )
    block_scales = (
        expanded_scales.view(output_size // 128, 128, input_size // 128, 128)
        .amax(dim=(1, 3), keepdim=True)
        / 64.0
    )
    expected_weight = (
        block_values
        * expanded_scales.view(output_size // 128, 128, input_size // 128, 128)
        / block_scales
    ).reshape(output_size, input_size)
    return expected_weight.to(torch.float8_e4m3fn), block_scales.squeeze((1, 3))


def _assert_rocm_conversion(weight, scale, packed, scale_bytes, shuffle_weight):
    expected_weight, expected_block_scale = _expected_fp8_conversion(
        packed, scale_bytes
    )
    expected_weight = expected_weight.view(torch.int8)
    expected_weight[expected_weight == -128] = 0
    expected_weight = expected_weight.view(torch.float8_e4m3fnuz)
    shuffled_weight = shuffle_weight(expected_weight.contiguous(), (16, 16))
    expected_scale = expected_block_scale * 2.0

    assert weight.dtype == torch.float8_e4m3fnuz
    assert scale.dtype == torch.float32
    assert torch.equal(weight.view(torch.int8), shuffled_weight.view(torch.int8))
    assert torch.equal(scale, expected_scale)

    reconstructed = expected_weight.float() * expected_scale.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    reference_values, reference_scales = _reference_fp4_dequant(packed, scale_bytes)
    assert torch.equal(reconstructed, reference_values * reference_scales)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not getattr(torch.version, "hip", None),
    reason="ROCm GPU is required",
)
@pytest.mark.parametrize(
    "num_experts,hidden_size,intermediate_size",
    [(2, 128, 128), (3, 256, 256)],
)
def test_rocm_dequant_fp4_to_fp8(
    monkeypatch, num_experts, hidden_size, intermediate_size
):
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    monkeypatch.setattr(fp8_quant, "_use_aiter", True)
    monkeypatch.setattr(fp8_quant, "_is_fp8_fnuz", True)

    layer = _make_layer(
        num_experts, 2 * intermediate_size, hidden_size, "cuda"
    )
    _add_w2(layer, num_experts, hidden_size, intermediate_size, "cuda")
    original_w13 = layer.w13_weight.data.clone()
    original_w13_scales = layer.w13_weight_scale_inv.data.clone()
    original_w2 = layer.w2_weight.data.clone()
    original_w2_scales = layer.w2_weight_scale_inv.data.clone()

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        is_fp4_experts=True,
    )
    config.dequant_fp4_to_fp8 = True
    Fp8MoEMethod(config).process_weights_after_loading_block_quant(layer)

    _assert_rocm_conversion(
        layer.w13_weight.data,
        layer.w13_weight_scale_inv.data,
        original_w13[0],
        original_w13_scales[0].view(torch.uint8),
        fp8_quant.shuffle_weight,
    )
    _assert_rocm_conversion(
        layer.w2_weight.data,
        layer.w2_weight_scale_inv.data,
        original_w2[0],
        original_w2_scales[0].view(torch.uint8),
        fp8_quant.shuffle_weight,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not getattr(torch.version, "hip", None),
    reason="ROCm GPU is required",
)
def test_rocm_native_fp4_default(monkeypatch):
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    monkeypatch.setattr(fp8_quant, "_use_aiter", True)
    monkeypatch.setattr(fp8_quant, "_is_fp8_fnuz", True)

    layer = _make_layer(2, 256, 128, "cuda")
    _add_w2(layer, 2, 128, 128, "cuda")
    original_w13 = layer.w13_weight.data.clone()
    original_w2 = layer.w2_weight.data.clone()

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        is_fp4_experts=True,
    )
    Fp8MoEMethod(config).process_weights_after_loading_block_quant(layer)

    assert layer.w13_weight.dtype == torch.float4_e2m1fn_x2
    assert layer.w2_weight.dtype == torch.float4_e2m1fn_x2
    assert layer.w13_weight.is_shuffled is False
    assert layer.w2_weight.is_shuffled is False
    assert torch.equal(layer.w13_weight.data.view(torch.int8), original_w13)
    assert torch.equal(layer.w2_weight.data.view(torch.int8), original_w2)


def test_non_rocm_dequant_fp4_to_fp8(monkeypatch):
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    monkeypatch.setattr(fp8_quant, "_use_aiter", False)
    monkeypatch.setattr(fp8_quant, "_is_fp8_fnuz", False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = _make_layer(2, 256, 128, device)
    _add_w2(layer, 2, 128, 128, device)
    original_w13 = layer.w13_weight.data.clone()
    original_w13_scales = layer.w13_weight_scale_inv.data.clone()

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        is_fp4_experts=True,
    )
    config.dequant_fp4_to_fp8 = True
    Fp8MoEMethod(config).process_weights_after_loading_block_quant(layer)

    expected_weight, expected_scale = _expected_fp8_conversion(
        original_w13[0], original_w13_scales[0].view(torch.uint8)
    )
    assert layer.w13_weight.dtype == torch.float8_e4m3fn
    assert layer.w13_weight_scale_inv.dtype == torch.float32
    assert torch.equal(
        layer.w13_weight.data.view(torch.int8),
        expected_weight.view(torch.int8),
    )
    assert torch.equal(layer.w13_weight_scale_inv.data, expected_scale)
