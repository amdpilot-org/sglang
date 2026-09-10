import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import (
    silu_and_mul_contig_post_quant,
    silu_and_mul_masked_post_quant,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-compatible GPU is required"
)

output_dtype = torch.uint8 if torch.version.hip is not None else fp8_dtype


def _activation_reference(input_tensor):
    hidden_dim = input_tensor.shape[-1] // 2
    gate = input_tensor[..., :hidden_dim].float()
    up = input_tensor[..., hidden_dim:].float()
    return torch.nn.functional.silu(gate) * up


def _quantization_reference(activation):
    groups = activation.unflatten(-1, (-1, 128))
    absmax = groups.abs().amax(-1).clamp_min(1e-10)
    raw_scale = absmax / fp8_max
    scale_bits = raw_scale.contiguous().view(torch.int32)
    exponent = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        torch.int32
    )
    scale = torch.exp2(exponent.float() - 127.0)
    quantized = (groups / scale[..., None]).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return quantized, scale


def _assert_quantization(output, output_scale, input_tensor):
    activation = _activation_reference(input_tensor)
    expected_quantized, expected_scale = _quantization_reference(activation)

    torch.testing.assert_close(output_scale, expected_scale, rtol=1e-5, atol=1e-8)

    fp8_output = output.view(fp8_dtype) if output.dtype == torch.uint8 else output
    dequantized = fp8_output.float().unflatten(-1, (-1, 128))
    dequantized = dequantized * output_scale[..., None]
    dequantized = dequantized.flatten(-2)
    torch.testing.assert_close(dequantized, activation, rtol=0.1, atol=0.1)


@requires_cuda
@pytest.mark.parametrize("hidden_dim", [7936, 8192])
def test_contiguous_supported_boundary_widths(hidden_dim):
    torch.manual_seed(hidden_dim)
    input_tensor = (
        torch.randn((33, hidden_dim * 2), dtype=torch.bfloat16, device="cuda") * 0.005
    )
    output = torch.empty((33, hidden_dim), dtype=output_dtype, device="cuda")
    output_scale = torch.empty(
        (33, hidden_dim // 128), dtype=torch.float32, device="cuda"
    )

    silu_and_mul_contig_post_quant(
        input_tensor, output, output_scale, 128, scale_ue8m0=True
    )
    torch.cuda.synchronize()

    _assert_quantization(output, output_scale, input_tensor)


@requires_cuda
def test_contiguous_zero_rows_is_noop():
    hidden_dim = 1024
    input_tensor = torch.empty((0, hidden_dim * 2), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((0, hidden_dim), dtype=output_dtype, device="cuda")
    output_scale = torch.empty(
        (0, hidden_dim // 128), dtype=torch.float32, device="cuda"
    )

    silu_and_mul_contig_post_quant(input_tensor, output, output_scale, 128)
    torch.cuda.synchronize()

    assert input_tensor.shape == (0, hidden_dim * 2)
    assert output.shape == (0, hidden_dim)
    assert output_scale.shape == (0, hidden_dim // 128)


@requires_cuda
def test_contiguous_width_above_thread_limit_refuses_launch():
    hidden_dim = 8448
    input_tensor = torch.empty((1, hidden_dim * 2), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((1, hidden_dim), dtype=output_dtype, device="cuda")
    output_scale = torch.empty(
        (1, hidden_dim // 128), dtype=torch.float32, device="cuda"
    )

    with pytest.raises(RuntimeError, match="hidden_dim/8 exceeds CUDA block limit"):
        silu_and_mul_contig_post_quant(input_tensor, output, output_scale, 128)


@requires_cuda
def test_masked_supported_shape_matches_reference():
    torch.manual_seed(0)
    num_experts, num_tokens, hidden_dim = 2, 5, 1024
    input_tensor = (
        torch.randn(
            (num_experts, num_tokens, hidden_dim * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.005
    )
    output = torch.full(
        (num_experts, num_tokens, hidden_dim),
        0x3F if output_dtype == torch.uint8 else 1.0,
        dtype=output_dtype,
        device="cuda",
    )
    output_scale = torch.full(
        (num_experts, num_tokens, hidden_dim // 128),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    masked_m = torch.tensor([2, 1], dtype=torch.int32, device="cuda")

    silu_and_mul_masked_post_quant(
        input_tensor,
        output,
        output_scale,
        128,
        masked_m,
        scale_ue8m0=True,
        topk=1,
    )
    torch.cuda.synchronize()

    valid = torch.tensor(
        [[True, True, False, False, False], [True, False, False, False, False]],
        device="cuda",
    )
    _assert_quantization(output[valid], output_scale[valid], input_tensor[valid])
    assert (output[~valid] == (0x3F if output_dtype == torch.uint8 else 1.0)).all()
    assert torch.isnan(output_scale[~valid]).all()


@requires_cuda
def test_masked_zero_work_is_noop():
    num_experts, num_tokens, hidden_dim = 2, 3, 1024
    input_tensor = torch.randn(
        (num_experts, num_tokens, hidden_dim * 2),
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.full(
        (num_experts, num_tokens, hidden_dim),
        0x3F if output_dtype == torch.uint8 else 1.0,
        dtype=output_dtype,
        device="cuda",
    )
    output_scale = torch.full(
        (num_experts, num_tokens, hidden_dim // 128),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    masked_m = torch.tensor([1, 2], dtype=torch.int32, device="cuda")

    silu_and_mul_masked_post_quant(
        input_tensor, output, output_scale, 128, masked_m, topk=0
    )
    torch.cuda.synchronize()

    assert (output == (0x3F if output_dtype == torch.uint8 else 1.0)).all()
    assert torch.isnan(output_scale).all()


@requires_cuda
def test_masked_width_above_thread_limit_refuses_launch():
    hidden_dim = 8448
    input_tensor = torch.empty(
        (2, 1, hidden_dim * 2), dtype=torch.bfloat16, device="cuda"
    )
    output = torch.empty((2, 1, hidden_dim), dtype=output_dtype, device="cuda")
    output_scale = torch.empty(
        (2, 1, hidden_dim // 128), dtype=torch.float32, device="cuda"
    )
    masked_m = torch.tensor([1, 1], dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="hidden_dim/8 exceeds CUDA block limit"):
        silu_and_mul_masked_post_quant(
            input_tensor, output, output_scale, 128, masked_m, topk=1
        )
