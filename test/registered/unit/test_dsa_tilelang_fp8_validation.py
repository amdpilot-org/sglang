"""Validation boundaries for CUDA TileLang DSA with an FP8 KV cache."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import _check_tilelang_dsa_fp8_kv
from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDsaTilelangFp8Validation(CustomTestCase):
    def test_cuda_fp8_mixed_decode_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "tilelang", hip=False)

    def test_cuda_fp8_mixed_prefill_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "trtllm", hip=False)

    def test_hip_fp8_tilelang_allowed(self):
        # ROCm has a real fp8 tilelang kernel
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=True)

    @patch("torch.cuda.get_device_capability", return_value=(9, 0))
    def test_cuda_sm90_fp8_tilelang_allowed(self, _capability):
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=False)

    @patch("torch.cuda.get_device_capability", return_value=(8, 0))
    def test_cuda_pre_sm89_fp8_tilelang_rejected(self, _capability):
        with self.assertRaisesRegex(ValueError, "SM89"):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_dcp_rejected(self):
        with self.assertRaisesRegex(ValueError, "dcp-size"):
            _check_tilelang_dsa_fp8_kv(
                "fp8_e4m3", "tilelang", "tilelang", hip=False, dcp_size=2
            )

    def test_bf16_tilelang_allowed(self):
        # what the CUDA kernel expects
        _check_tilelang_dsa_fp8_kv("bfloat16", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_non_tilelang_allowed(self):
        # fp8-capable backends must pass
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "trtllm", hip=False)

    def test_cuda_tilelang_fp8_selects_raw_layout(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(), kv_lora_rank=512, qk_rope_head_dim=0
        )
        execution = SimpleNamespace(
            kernel=SimpleNamespace(
                dsa_prefill_backend="tilelang", dsa_decode_backend="tilelang"
            )
        )
        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.is_deepseek_dsa",
                return_value=True,
            ),
            patch("sglang.srt.mem_cache.kv_cache_configurator._is_hip", False),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_exec",
                return_value=execution,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=None),
            ),
        ):
            self.assertEqual(
                calculate_mla_kv_cache_dim(
                    model_config=model_config, kv_cache_dtype=torch.float8_e4m3fn
                ),
                512,
            )

    def test_cuda_non_tilelang_fp8_keeps_scaled_layout(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(), kv_lora_rank=512, qk_rope_head_dim=0
        )
        execution = SimpleNamespace(
            kernel=SimpleNamespace(
                dsa_prefill_backend="flashmla_kv", dsa_decode_backend="flashmla_kv"
            )
        )
        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.is_deepseek_dsa",
                return_value=True,
            ),
            patch("sglang.srt.mem_cache.kv_cache_configurator._is_hip", False),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_exec",
                return_value=execution,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=None),
            ),
        ):
            self.assertEqual(
                calculate_mla_kv_cache_dim(
                    model_config=model_config, kv_cache_dtype=torch.float8_e4m3fn
                ),
                528,
            )


if __name__ == "__main__":
    unittest.main()
