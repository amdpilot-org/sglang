"""CPU unit tests for MXFP4 conversion and MXFP8 fake-output metadata."""

import unittest
from unittest import mock

import torch

import sglang.srt.layers.quantization.fp8_utils as fp8_utils

from sglang.srt.layers.quantization.fp8_utils import (
    _fake_flashinfer_mxfp8_quantize,
    quantize_block_fp8_weight_to_mxfp4,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestFp8UtilsMxfp4(CustomTestCase):
    def test_fake_flashinfer_mxfp8_quantize_linear_scale_shape(self):
        """The fake op must flatten leading dimensions and preserve scale groups."""
        input = torch.empty((2, 3, 96), dtype=torch.bfloat16)

        quantized, scale = _fake_flashinfer_mxfp8_quantize(input, False, alignment=128)

        self.assertEqual(quantized.shape, torch.Size([6, 128]))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, torch.Size([24]))
        self.assertEqual(scale.dtype, torch.uint8)

    def test_fake_flashinfer_mxfp8_quantize_swizzled_scale_shape(self):
        input = torch.empty((3, 64), dtype=torch.bfloat16)

        quantized, scale = _fake_flashinfer_mxfp8_quantize(input, True, alignment=64)

        self.assertEqual(quantized.shape, torch.Size([3, 64]))
        self.assertEqual(scale.shape, torch.Size([512]))

    def test_quantize_block_fp8_weight_to_mxfp4_shapes_and_dtype(self):
        fp8_weight = (
            torch.linspace(-2.0, 2.0, 32 * 32, dtype=torch.float32)
            .reshape(32, 32)
            .to(torch.float8_e4m3fn)
        )
        fp8_scale = torch.ones(1, 1, dtype=torch.float8_e8m0fnu)

        fp4_weight, fp4_scale = quantize_block_fp8_weight_to_mxfp4(
            fp8_weight, fp8_scale, [128, 128]
        )

        self.assertEqual(fp4_weight.dtype, torch.int8)
        self.assertEqual(fp4_weight.shape, torch.Size([32, 16]))
        self.assertEqual(fp4_scale.dtype, torch.float8_e8m0fnu)
        self.assertEqual(fp4_scale.shape, torch.Size([32, 1]))

    def test_quantize_block_fp8_weight_to_mxfp4_grouped_weight(self):
        fp8_weight = (
            torch.linspace(-2.0, 2.0, 2 * 32 * 32, dtype=torch.float32)
            .reshape(2, 32, 32)
            .to(torch.float8_e4m3fn)
        )
        fp8_scale = torch.ones(2, 1, 1, dtype=torch.float8_e8m0fnu)

        fp4_weight, fp4_scale = quantize_block_fp8_weight_to_mxfp4(
            fp8_weight, fp8_scale, [128, 128]
        )

        self.assertEqual(fp4_weight.dtype, torch.int8)
        self.assertEqual(fp4_weight.shape, torch.Size([2, 32, 16]))
        self.assertEqual(fp4_scale.dtype, torch.float8_e8m0fnu)
        self.assertEqual(fp4_scale.shape, torch.Size([2, 32, 1]))

    def test_quantize_block_fp8_weight_to_mxfp4_chunks_on_scale_rows(self):
        """Large shared experts must not be fully dequantized at once."""
        torch.manual_seed(0)
        fp8_weight = torch.randn(2, 257, 128).to(torch.float8_e4m3fn)
        fp8_scale = torch.rand(2, 3, 1, dtype=torch.float32) + 0.5

        with mock.patch.object(fp8_utils, "_FP8_TO_MXFP4_CHUNK_ELEMENTS", 128 * 128):
            chunked_weight, chunked_scale = quantize_block_fp8_weight_to_mxfp4(
                fp8_weight, fp8_scale, [128, 128]
            )
        with mock.patch.object(fp8_utils, "_FP8_TO_MXFP4_CHUNK_ELEMENTS", 1 << 60):
            reference_weight, reference_scale = quantize_block_fp8_weight_to_mxfp4(
                fp8_weight, fp8_scale, [128, 128]
            )

        # Covers grouped input, multiple chunks, and a one-row final FP8 block.
        torch.testing.assert_close(chunked_weight, reference_weight)
        torch.testing.assert_close(chunked_scale, reference_scale)

    def test_quantize_block_fp8_weight_to_mxfp4_rejects_partial_output_group(self):
        fp8_weight = torch.ones(4, 48, dtype=torch.float8_e4m3fn)
        fp8_scale = torch.ones(1, 1, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "K divisible by 32"):
            quantize_block_fp8_weight_to_mxfp4(
                fp8_weight, fp8_scale, [128, 128]
            )


if __name__ == "__main__":
    unittest.main()
