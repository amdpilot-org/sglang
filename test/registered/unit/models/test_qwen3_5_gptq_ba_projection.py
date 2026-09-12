import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet


class _QuantConfig:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class TestQwen35GptqBaProjection(unittest.TestCase):
    def _create(self, quant_name, num_v_heads=48, tp_size=1):
        quant_config = None if quant_name is None else _QuantConfig(quant_name)
        sentinel = object()
        with patch(
            "sglang.srt.models.qwen3_5.MergedColumnParallelLinear",
            return_value=sentinel,
        ) as linear:
            result = Qwen3_5GatedDeltaNet.create_ba_proj(
                object(),
                hidden_size=5120,
                num_v_heads=num_v_heads,
                quant_config=quant_config,
                prefix="model.layers.0.linear_attn.in_proj_ba",
                tp_rank=0,
                tp_size=tp_size,
            )
        self.assertIs(result, sentinel)
        return linear.call_args.kwargs["quant_config"], quant_config

    def test_gptq_marlin_incompatible_width_is_unquantized(self):
        actual, _ = self._create("gptq_marlin")
        self.assertIsNone(actual)

    def test_gptq_marlin_compatible_width_remains_quantized(self):
        actual, expected = self._create("gptq_marlin", num_v_heads=64)
        self.assertIs(actual, expected)

    def test_tensor_parallel_local_width_controls_compatibility(self):
        actual, _ = self._create("gptq_marlin", num_v_heads=48, tp_size=2)
        self.assertIsNone(actual)

    def test_other_quantizers_are_unchanged(self):
        actual, expected = self._create("fp8")
        self.assertIs(actual, expected)

    def test_unquantized_config_is_unchanged(self):
        actual, expected = self._create(None)
        self.assertIs(actual, expected)


if __name__ == "__main__":
    unittest.main()
