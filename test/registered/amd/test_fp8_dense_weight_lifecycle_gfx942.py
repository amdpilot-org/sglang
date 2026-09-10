"""Dense FP8 weight lifecycle coverage for MI300X (gfx942).

The test deliberately keeps AWQ/GPTQ packing, INT8, MoE, and MTP selection out of
scope.  It loads a small checkpoint-format block-FP8 dense weight, exercises the
real post-load transform, and executes the dispatched Triton linear kernel.
"""

import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.layer_ut_utils import (
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)

register_amd_ci(est_time=10, suite="stage-a-test-1-gpu-small-amd")

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _is_gfx942():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 4)


def _known_source_tensor(rows, columns, modulus, device):
    values = torch.arange(rows * columns, device=device, dtype=torch.float32)
    values = ((values % modulus) - (modulus // 2)) / (modulus // 2)
    return values.reshape(rows, columns).clamp(-1.0, 1.0)


def _quantize_blockwise(weight, block=128):
    output_size, input_size = weight.shape
    tiles = weight.float().reshape(
        output_size // block, block, input_size // block, block
    )
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / FP8_MAX
    quantized = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scale[:, None, :, None]).reshape(
        output_size, input_size
    )
    return quantized.reshape(output_size, input_size), scale, dequantized


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class TestFp8DenseWeightLifecycleGfx942(unittest.TestCase):
    def setUp(self):
        self._parallel_override = None
        if not _is_gfx942():
            self.skipTest("dense FP8 lifecycle test requires gfx942")
        init_single_process_dist()
        self._parallel_override = get_parallel().override(tp_size=1)
        self._parallel_override.__enter__()

    def tearDown(self):
        if self._parallel_override is not None:
            self._parallel_override.__exit__(None, None, None)

    def test_checkpoint_inverse_scale_survives_post_load_and_kernel(self):
        output_size, input_size, tokens = 128, 256, 64
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        layer = make_tp1_column_parallel_linear(
            quant_config, output_size, input_size
        )

        source_weight = _known_source_tensor(
            output_size, input_size, modulus=19, device="cuda"
        )
        checkpoint_weight, checkpoint_scale, dequantized_weight = _quantize_blockwise(
            source_weight
        )
        load_linear_weights(
            layer,
            weight=checkpoint_weight,
            weight_scale_inv=checkpoint_scale,
        )

        checkpoint_scale_copy = checkpoint_scale.detach().clone()
        layer.quant_method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fnuz)
        self.assertIsNone(layer.input_scale)
        self.assertTrue(
            torch.equal(layer.weight_scale_inv.detach(), checkpoint_scale_copy * 2.0)
        )
        self.assertTrue(
            torch.equal(
                layer.weight.detach().view(torch.int8),
                checkpoint_weight.detach().view(torch.int8),
            )
        )

        source_input = _known_source_tensor(
            tokens, input_size, modulus=17, device="cuda"
        ).to(torch.bfloat16)
        quantized_input, input_scale = per_token_group_quant_fp8(
            source_input, group_size=128, column_major_scales=False
        )
        input_scale_copy = input_scale.detach().clone()
        output, _ = layer((quantized_input, input_scale))
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(input_scale, input_scale_copy))
        dequantized_input = quantized_input.float() * input_scale.repeat_interleave(
            128, dim=1
        )
        reference = dequantized_input @ dequantized_weight.t()
        torch.testing.assert_close(
            output.float(),
            reference,
            rtol=5e-2,
            atol=1e-1,
        )


if __name__ == "__main__":
    unittest.main()
