"""Compressed-tensors NEXTN/MTP scheme-selection regressions."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.test_utils import CustomTestCase

MTP_LAYER = "mtp.layers.0.self_attn.qkv_proj"
PACKED_WEIGHT_NAMES = {
    f"{MTP_LAYER}.weight_packed",
    f"{MTP_LAYER}.weight_scale",
}

W4A8_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "token",
                "dynamic": True,
            },
        }
    },
    "ignore": [],
}


def _config(ignore=()):
    return CompressedTensorsConfig.from_config({**W4A8_CONFIG, "ignore": list(ignore)})


def _linear(config):
    return ReplicatedLinear(
        input_size=8,
        output_size=16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix=MTP_LAYER,
    )


class TestCompressedTensorsNextN(CustomTestCase):
    def test_explicit_ignore_selects_dense_linear(self):
        layer = _linear(_config(["re:mtp\\..*"]))

        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)

    def test_documented_regex_covers_both_mtp_prefix_forms(self):
        ignore = ["re:(model\\.)?mtp\\..*"]

        for prefix in (MTP_LAYER, f"model.{MTP_LAYER}"):
            with self.subTest(prefix=prefix):
                layer = ReplicatedLinear(
                    input_size=8,
                    output_size=16,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=_config(ignore),
                    prefix=prefix,
                )
                self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_plain_glob_looking_ignores_do_not_match_mtp_children(self):
        for ignore in (["mtp.*"], ["model.mtp.*"]):
            with self.subTest(ignore=ignore):
                with self.assertRaises(NotImplementedError):
                    _linear(_config(ignore))

    def test_dense_mtp_checkpoint_has_actionable_construction_error(self):
        # Dense checkpoint storage is not visible while modules are constructed.
        # The unsupported broad Linear match must therefore fail with enough
        # context to add a correct checkpoint-local ignore entry.
        dense_weight_names = {f"{MTP_LAYER}.weight"}
        self.assertNotEqual(dense_weight_names, PACKED_WEIGHT_NAMES)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"mtp\.layers\.0\.self_attn\.qkv_proj.*matched target 'Linear'.*"
            r"format='pack-quantized'.*weights=.*num_bits=4.*"
            r"input_activations=.*num_bits=8.*quantization_config\.ignore",
        ):
            _linear(_config())

    def test_packed_mtp_checkpoint_is_not_silently_dequantized(self):
        # The same pre-load selection is also reached by checkpoints that really
        # carry packed MTP tensors. Do not infer a dense fallback from the prefix.
        self.assertIn(f"{MTP_LAYER}.weight_packed", PACKED_WEIGHT_NAMES)
        with patch.object(
            UnquantizedLinearMethod,
            "create_weights",
            side_effect=AssertionError("packed MTP was silently dequantized"),
        ):
            with self.assertRaises(NotImplementedError):
                _linear(_config())


if __name__ == "__main__":
    unittest.main()
