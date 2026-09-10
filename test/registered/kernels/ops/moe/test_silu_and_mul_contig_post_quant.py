"""Contracts for the contiguous fused SiLU-and-mul FP8 quantization kernel."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.moe import silu_and_mul_contig_post_quant
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _fp8_storage_dtype():
    if torch.version.hip and torch.cuda.get_device_capability() == (9, 4):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _fp8_max():
    return 224.0 if _fp8_storage_dtype() == torch.float8_e4m3fnuz else 448.0


class TestSiluAndMulContigPostQuant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_bf16_fp8_contract_preserves_addresses_and_sentinel(self):
        num_tokens = 17
        hidden_dim = 768
        num_groups = hidden_dim // 128
        generator = torch.Generator(device="cuda").manual_seed(32065)
        gate_up = (
            torch.randn(
                (num_tokens, 2 * hidden_dim),
                generator=generator,
                device="cuda",
            )
            * 3.0
        ).to(torch.bfloat16)

        output_storage = torch.empty(
            (num_tokens + 1, hidden_dim), device="cuda", dtype=torch.float8_e4m3fn
        )
        scale_storage = torch.empty(
            (num_tokens + 1, num_groups), device="cuda", dtype=torch.float32
        )
        output_storage.view(torch.uint8).fill_(0x7F)
        scale_storage.fill_(float("nan"))
        output = output_storage[:num_tokens]
        output_scale = scale_storage[:num_tokens]
        addresses = (gate_up.data_ptr(), output.data_ptr(), output_scale.data_ptr())

        silu_and_mul_contig_post_quant(
            gate_up,
            output,
            output_scale,
            128,
            scale_ue8m0=False,
            transposed=False,
        )
        torch.cuda.synchronize()

        self.assertEqual(
            addresses,
            (gate_up.data_ptr(), output.data_ptr(), output_scale.data_ptr()),
        )
        self.assertTrue(
            bool((output_storage[num_tokens:].view(torch.uint8) == 0x7F).all().item())
        )
        self.assertTrue(bool(scale_storage[num_tokens:].isnan().all().item()))

        gate, up = gate_up.chunk(2, dim=-1)
        reference = gate.float() * torch.sigmoid(gate.float()) * up.float()
        grouped_reference = reference.view(num_tokens, num_groups, 128)
        expected_scale = (
            grouped_reference.abs().amax(dim=-1).clamp_min(1e-10) / _fp8_max()
        )
        torch.testing.assert_close(
            output_scale,
            expected_scale,
            rtol=5e-3,
            atol=1e-6,
        )

        expanded_scale = output_scale.repeat_interleave(128, dim=-1)
        dequantized = output.view(_fp8_storage_dtype()).float() * expanded_scale
        self.assertTrue(
            bool(
                (
                    (dequantized - reference).abs()
                    <= expanded_scale * 16.0 + 1e-4
                ).all().item()
            )
        )

    def test_zero_tokens_is_valid_noop(self):
        gate_up = torch.empty((0, 1536), device="cuda", dtype=torch.bfloat16)
        output = torch.empty((0, 768), device="cuda", dtype=torch.float8_e4m3fn)
        output_scale = torch.empty((0, 6), device="cuda", dtype=torch.float32)

        silu_and_mul_contig_post_quant(
            gate_up,
            output,
            output_scale,
            128,
            scale_ue8m0=False,
            transposed=False,
        )
        torch.cuda.synchronize()

    def test_unsupported_width_fails_before_launch(self):
        hidden_dim = 8320
        gate_up = torch.empty(
            (1, 2 * hidden_dim), device="cuda", dtype=torch.bfloat16
        )
        output = torch.empty(
            (1, hidden_dim), device="cuda", dtype=torch.float8_e4m3fn
        )
        output.view(torch.uint8).fill_(0x7F)
        output_scale = torch.empty(
            (1, hidden_dim // 128), device="cuda", dtype=torch.float32
        )

        with self.assertRaisesRegex(RuntimeError, "1024 threads per block"):
            silu_and_mul_contig_post_quant(
                gate_up,
                output,
                output_scale,
                128,
                scale_ue8m0=False,
                transposed=False,
            )
        self.assertTrue(bool((output.view(torch.uint8) == 0x7F).all().item()))

    def test_unsupported_fp16_input_fails_before_launch(self):
        gate_up = torch.empty((1, 1536), device="cuda", dtype=torch.float16)
        output = torch.empty((1, 768), device="cuda", dtype=torch.float8_e4m3fn)
        output.view(torch.uint8).fill_(0x7F)
        output_scale = torch.empty((1, 6), device="cuda", dtype=torch.float32)

        with self.assertRaises(RuntimeError):
            silu_and_mul_contig_post_quant(
                gate_up,
                output,
                output_scale,
                128,
                scale_ue8m0=False,
                transposed=False,
            )
        self.assertTrue(bool((output.view(torch.uint8) == 0x7F).all().item()))


if __name__ == "__main__":
    unittest.main()
