"""Structured-input coverage for the gfx942 INT8 block GEMM.

The existing INT8 tests primarily use random operands. These cases keep the
packed INT8 representation and [128, 128] scales fixed while varying complete
activation blocks between zero/tiny values, mixed magnitudes, and cancellation
or skewed sparse states.
"""

import unittest

import torch

from sglang.kernels.ops.quantization.int8_kernel import w8a8_block_int8_matmul
from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

N = 1024
K = 1024
BLOCK = 128
OUTPUT_DTYPE = torch.bfloat16


def _make_weight(device):
    torch.manual_seed(15194)
    weight = torch.randint(-127, 128, (N, K), device=device, dtype=torch.int8)
    weight[:, :BLOCK] = 1
    scales = torch.logspace(
        -6,
        -2,
        steps=(N // BLOCK) * (K // BLOCK),
        device=device,
        dtype=torch.float32,
    ).reshape(N // BLOCK, K // BLOCK)
    return weight, scales


def _make_activation(name, m, device):
    if name == "zero_tiny":
        activation = torch.zeros((m, K), device=device, dtype=torch.int8)
        tiny = torch.rand((m, K), device=device) < 0.1
        activation[tiny] = torch.where(
            torch.rand((m, K), device=device)[tiny] < 0.5, -1, 1
        ).to(torch.int8)
        scales = torch.full((m, K // BLOCK), 1e-6, device=device)
    elif name == "mixed_magnitude":
        activation = torch.randint(-127, 128, (m, K), device=device, dtype=torch.int8)
        scales = torch.logspace(
            -6, -2, steps=m * (K // BLOCK), device=device, dtype=torch.float32
        ).reshape(m, K // BLOCK)
    elif name == "cancellation_skewed":
        activation = torch.zeros((m, K), device=device, dtype=torch.int8)
        alternating = torch.arange(m, device=device) % 4 == 0
        activation[alternating, 0::2] = 127
        activation[alternating, 1::2] = -127
        skewed = ~alternating
        nonzero = torch.rand((m, K), device=device) < 0.1
        signs = torch.where(torch.rand((m, K), device=device) < 0.5, -127, 127).to(
            torch.int8
        )
        activation[skewed] = torch.where(
            nonzero, signs, torch.zeros_like(signs)
        )[skewed]
        scales = torch.full((m, K // BLOCK), 1e-3, device=device)
    else:
        raise ValueError(f"unknown input distribution: {name}")
    return activation, scales


def _dequantized_reference(activation, activation_scales, weight, weight_scales):
    activation_float = activation.to(torch.float32) * activation_scales.repeat_interleave(
        BLOCK, dim=1
    )
    weight_float = weight.to(torch.float32) * weight_scales.repeat_interleave(
        BLOCK, dim=0
    ).repeat_interleave(BLOCK, dim=1)
    return (activation_float @ weight_float.T).to(OUTPUT_DTYPE)


@unittest.skipUnless(is_gfx942_supported(), "requires AMD gfx942")
class TestInt8BlockGemmStructuredInputs(CustomTestCase):
    def test_structured_inputs(self):
        device = torch.device("cuda")
        weight, weight_scales = _make_weight(device)
        for m in (1, 4096):
            for name in ("zero_tiny", "mixed_magnitude", "cancellation_skewed"):
                with self.subTest(m=m, input_distribution=name):
                    activation, activation_scales = _make_activation(name, m, device)
                    actual = w8a8_block_int8_matmul(
                        activation,
                        weight,
                        activation_scales,
                        weight_scales,
                        [BLOCK, BLOCK],
                        OUTPUT_DTYPE,
                    )
                    reference = _dequantized_reference(
                        activation,
                        activation_scales,
                        weight,
                        weight_scales,
                    )

                    self.assertTrue(torch.isfinite(actual).all())
                    difference = (
                        actual.to(torch.float32) - reference.to(torch.float32)
                    ).abs()
                    reference_mean = reference.to(torch.float32).abs().mean()
                    relative_error = difference.mean() / reference_mean.clamp(min=1e-30)
                    self.assertLess(relative_error.item(), 0.02)


if __name__ == "__main__":
    unittest.main()
