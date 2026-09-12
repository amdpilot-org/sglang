import unittest

from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.glm5_next import can_fuse_glm5_next_qkvbfg
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


PREFIX = "model.layers.0.linear_attn"
PROJECTION_NAMES = (
    "qkv_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
    "g_a_proj",
    "g_b_proj",
)


class TestGlm5NextQkvbfgFusion(CustomTestCase):
    def test_fp8_experts_with_unquantized_attention_can_fuse(self):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            ignored_layers=[PREFIX],
        )

        self.assertTrue(
            can_fuse_glm5_next_qkvbfg(
                quant_config, PREFIX, head_shard_size=8, tp_size=8
            )
        )

    def test_partially_quantized_attention_cannot_fuse(self):
        ignored_layers = [
            f"{PREFIX}.{name}" for name in PROJECTION_NAMES if name != "g_b_proj"
        ]
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            ignored_layers=ignored_layers,
        )

        self.assertFalse(
            can_fuse_glm5_next_qkvbfg(
                quant_config, PREFIX, head_shard_size=8, tp_size=8
            )
        )

    def test_attention_tp_mismatch_cannot_fuse(self):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            ignored_layers=[PREFIX],
        )

        self.assertFalse(
            can_fuse_glm5_next_qkvbfg(
                quant_config, PREFIX, head_shard_size=4, tp_size=8
            )
        )


if __name__ == "__main__":
    unittest.main()
