"""Compressed-tensors NEXTN selection and unquantized BF16 linear path."""

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4AFP8MoE,
)
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

NEXTN_LINEAR = "mtp.layers.0.self_attn.qkv_proj"
MOE_LAYER = "model.layers.0.mlp.experts"

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


def _config(ignore):
    return CompressedTensorsConfig.from_config({**W4A8_CONFIG, "ignore": ignore})


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip(), "requires one HIP-compatible GPU"
)
class TestCompressedTensorsNextNW4A8(CustomTestCase):
    def test_moe_group_selects_w4afp8_scheme(self):
        scheme = _config([]).get_moe_scheme(
            layer=FusedMoE.__new__(FusedMoE), layer_name=MOE_LAYER
        )

        self.assertIsInstance(scheme, CompressedTensorsW4AFP8MoE)

    def test_unignored_nextn_linear_error_names_module(self):
        with self.assertRaises(NotImplementedError) as context:
            ReplicatedLinear(
                input_size=8,
                output_size=16,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=_config([]),
                prefix=NEXTN_LINEAR,
            )

        message = str(context.exception)
        self.assertIn(repr(NEXTN_LINEAR), message)
        self.assertIn("quantization_config.ignore", message)

    def test_ignored_nextn_bf16_weights_run_gpu_linear(self):
        torch.manual_seed(38574)
        layer = ReplicatedLinear(
            input_size=8,
            output_size=16,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_config(["re:mtp\\..*"]),
            prefix=NEXTN_LINEAR,
        ).cuda()
        weight = torch.nn.Parameter(
            torch.randn(16, 8, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.weight = weight
        inputs = torch.randn(3, 8, device="cuda", dtype=torch.bfloat16)

        output = layer(inputs)[0]

        self.assertEqual(type(layer.quant_method).__name__, "UnquantizedLinearMethod")
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output, F.linear(inputs, weight), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
