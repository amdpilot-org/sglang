"""Real-path gfx942 validation for the Triton block-FP8 GEMM.

The existing gfx95 tests cover the CK bpreshuffle path. On gfx942,
``aiter_w8a8_block_fp8_linear`` instead routes to AITER's Triton block-FP8 GEMM.
These tests validate that uncovered execution representation with independently
dequantized inputs, a sentinel-filled output buffer, BF16/FP16 output dtypes, and
one bounded CUDA-graph replay.
"""

import json
import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.utils.common import is_gfx942_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

_N, _K = 512, 7168
_BLOCK_N, _BLOCK_K = 128, 128
_MS = (1, 128, 1024)


@unittest.skipUnless(
    is_hip() and is_gfx942_supported() and fp8_utils._use_aiter,
    "block-FP8 Triton GEMM representation is a gfx942 + aiter path",
)
class TestGfx942BlockFP8TritonGEMM(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda"
        cls.fp8_dtype = fp8_utils.aiter.dtypes.fp8

    def setUp(self):
        torch.manual_seed(36390)
        torch.cuda.manual_seed_all(36390)

    def _make_case(self, m, output_dtype):
        q_float = torch.rand(m, _K, device=self.device, dtype=torch.float32) * 2 - 1
        w_float = torch.rand(_N, _K, device=self.device, dtype=torch.float32) * 2 - 1
        q_input = q_float.to(self.fp8_dtype)
        weight = w_float.to(self.fp8_dtype)
        input_scale = (
            torch.rand(
                m,
                _K // _BLOCK_K,
                device=self.device,
                dtype=torch.float32,
            )
            + 0.5
        ) * 1e-3
        weight_scale = (
            torch.rand(
                _N // _BLOCK_N,
                _K // _BLOCK_K,
                device=self.device,
                dtype=torch.float32,
            )
            + 0.5
        ) * 1e-3
        output = torch.full(
            (m, _N), float("nan"), device=self.device, dtype=output_dtype
        )
        return q_input, weight, input_scale, weight_scale, output

    def _reference(self, q_input, weight, input_scale, weight_scale, output_dtype):
        expanded_input_scale = input_scale.repeat_interleave(_BLOCK_K, dim=1)
        expanded_weight_scale = weight_scale.repeat_interleave(
            _BLOCK_N, dim=0
        ).repeat_interleave(_BLOCK_K, dim=1)
        dequantized_input = q_input.to(torch.float32) * expanded_input_scale
        dequantized_weight = weight.to(torch.float32) * expanded_weight_scale
        return torch.matmul(dequantized_input, dequantized_weight.t()).to(output_dtype)

    def _timed_gemm(self, q_input, weight, input_scale, weight_scale, output_dtype):
        for _ in range(3):
            fp8_utils.triton_gemm_a8w8_blockscale(
                q_input,
                weight,
                input_scale,
                weight_scale,
                dtype=output_dtype,
            )
        torch.cuda.synchronize()

        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fp8_utils.triton_gemm_a8w8_blockscale(
                q_input,
                weight,
                input_scale,
                weight_scale,
                dtype=output_dtype,
            )
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        return samples

    def test_bf16_and_fp16_outputs_match_independent_reference(self):
        records = []
        for output_dtype in (torch.bfloat16, torch.float16):
            for m in _MS:
                with self.subTest(output_dtype=output_dtype, m=m):
                    case = self._make_case(m, output_dtype)
                    q_input, weight, input_scale, weight_scale, output = case
                    result = fp8_utils.triton_gemm_a8w8_blockscale(
                        q_input,
                        weight,
                        input_scale,
                        weight_scale,
                        dtype=output_dtype,
                        y=output,
                    )
                    torch.cuda.synchronize()
                    reference = self._reference(
                        q_input, weight, input_scale, weight_scale, output_dtype
                    )
                    difference = (
                        result.to(torch.float32) - reference.to(torch.float32)
                    ).abs()

                    self.assertIs(result, output)
                    self.assertEqual(result.data_ptr(), output.data_ptr())
                    self.assertNotEqual(result.data_ptr(), q_input.data_ptr())
                    self.assertNotEqual(result.data_ptr(), weight.data_ptr())
                    self.assertFalse(bool(torch.isnan(result).any()))
                    self.assertTrue(bool(torch.isfinite(result).all()))
                    torch.testing.assert_close(
                        result,
                        reference,
                        rtol=2e-2,
                        atol=2e-5,
                    )

                    samples = self._timed_gemm(
                        q_input, weight, input_scale, weight_scale, output_dtype
                    )
                    records.append(
                        {
                            "m": m,
                            "output_dtype": str(output_dtype),
                            "mean_abs_error": difference.mean().item(),
                            "max_abs_error": difference.max().item(),
                            "timing_ms": {
                                "median": sorted(samples)[len(samples) // 2],
                                "samples": samples,
                            },
                        }
                    )
        print(json.dumps(records, sort_keys=True), flush=True)

    def test_wrapper_dispatches_prequantized_gfx942_path_to_triton(self):
        m = 128
        q_input, weight, input_scale, weight_scale, _ = self._make_case(
            m, torch.bfloat16
        )
        reference = self._reference(
            q_input, weight, input_scale, weight_scale, torch.bfloat16
        )
        with mock.patch.object(
            fp8_utils,
            "triton_gemm_a8w8_blockscale",
            wraps=fp8_utils.triton_gemm_a8w8_blockscale,
        ) as spy:
            result = fp8_utils.aiter_w8a8_block_fp8_linear(
                q_input,
                weight,
                [_BLOCK_N, _BLOCK_K],
                weight_scale,
                input_scale=input_scale,
            )
        spy.assert_called_once()
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(
            result, reference, rtol=2e-2, atol=2e-5
        )

    def test_static_output_address_survives_cuda_graph_replay(self):
        m = 128
        q_input, weight, input_scale, weight_scale, output = self._make_case(
            m, torch.bfloat16
        )
        fp8_utils.triton_gemm_a8w8_blockscale(
            q_input,
            weight,
            input_scale,
            weight_scale,
            dtype=torch.bfloat16,
            y=output,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fp8_utils.triton_gemm_a8w8_blockscale(
                q_input,
                weight,
                input_scale,
                weight_scale,
                dtype=torch.bfloat16,
                y=output,
            )
        output_address = output.data_ptr()

        replacement = (
            torch.rand(m, _K, device=self.device, dtype=torch.float32) * 2 - 1
        ).to(self.fp8_dtype)
        q_input.copy_(replacement)
        graph.replay()
        torch.cuda.synchronize()
        expected = fp8_utils.triton_gemm_a8w8_blockscale(
            replacement,
            weight,
            input_scale,
            weight_scale,
            dtype=torch.bfloat16,
        )
        self.assertEqual(output.data_ptr(), output_address)
        self.assertTrue(torch.equal(output, expected))

    def test_unsupported_input_dtype_fails_clearly(self):
        m = 1
        _, weight, _, weight_scale, _ = self._make_case(m, torch.bfloat16)
        input_tensor = torch.randn(
            m, _K, device=self.device, dtype=torch.float32
        )
        prequantized_tensor = torch.zeros(
            m, _K, device=self.device, dtype=torch.float16
        )
        input_scale = torch.ones(
            m, _K // _BLOCK_K, device=self.device, dtype=torch.float32
        )
        with self.assertRaisesRegex(ValueError, "bfloat16 and float16"):
            fp8_utils.aiter_w8a8_block_fp8_linear(
                input_tensor,
                weight,
                [_BLOCK_N, _BLOCK_K],
                weight_scale,
            )
        with self.assertRaisesRegex(ValueError, "native fp8 input dtype"):
            fp8_utils.aiter_w8a8_block_fp8_linear(
                prequantized_tensor,
                weight,
                [_BLOCK_N, _BLOCK_K],
                weight_scale,
                input_scale=input_scale,
            )


if __name__ == "__main__":
    unittest.main()
