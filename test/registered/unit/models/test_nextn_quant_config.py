from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.models.deepseek_nextn import (
    DeepseekModelNextN,
    DeepseekV3ForCausalLMNextN,
)
from sglang.srt.models.glm5_next_nextn import (
    Glm5NextForConditionalGenerationNextN,
)
from sglang.test.test_utils import CustomTestCase


class _QuantConfig:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class TestNextNQuantConfig(CustomTestCase):
    def test_glm_modelopt_fp4_reaches_nextn_decoder(self):
        config = SimpleNamespace(
            vocab_size=16,
            hidden_size=8,
            rms_norm_eps=1e-6,
            qk_rope_head_dim=0,
        )
        quant_config = _QuantConfig("modelopt_fp4")
        decoder = MagicMock()

        with (
            patch(
                "sglang.srt.models.deepseek_nextn.enable_nextn_moe_bf16_cast_to_fp8",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.get_embedding_tp_kwargs",
                return_value={},
            ),
            patch("sglang.srt.models.deepseek_nextn.VocabParallelEmbedding"),
            patch("sglang.srt.models.deepseek_nextn.RMSNorm"),
            patch("sglang.srt.models.deepseek_nextn.nn.Linear"),
            patch(
                "sglang.srt.models.deepseek_nextn.DeepseekV2DecoderLayer",
                decoder,
            ),
        ):
            DeepseekModelNextN(config, quant_config=quant_config)

        self.assertIs(decoder.call_args.kwargs["quant_config"], quant_config)

    def test_deepseek_modelopt_fp4_still_uses_bf16_nextn(self):
        quant_config = _QuantConfig("modelopt_fp4")
        resolved = DeepseekV3ForCausalLMNextN._resolve_nextn_quant_config(
            object(), SimpleNamespace(), quant_config
        )
        self.assertIsNone(resolved)

    def test_glm_modelopt_fp4_keeps_quantized_nextn(self):
        quant_config = _QuantConfig("modelopt_fp4")
        config = SimpleNamespace(num_hidden_layers=45, quantization_config={})
        resolved = Glm5NextForConditionalGenerationNextN._resolve_nextn_quant_config(
            object(), config, quant_config
        )
        self.assertIs(resolved, quant_config)

    def test_glm_checkpoint_excluded_nextn_still_uses_bf16(self):
        quant_config = _QuantConfig("modelopt_fp4")
        config = SimpleNamespace(
            num_hidden_layers=45,
            quantization_config={"ignore": ["model.layers.45.*"]},
        )
        resolved = Glm5NextForConditionalGenerationNextN._resolve_nextn_quant_config(
            object(), config, quant_config
        )
        self.assertIsNone(resolved)

    def test_non_modelopt_quantization_is_unchanged(self):
        quant_config = _QuantConfig("fp8")
        resolved = DeepseekV3ForCausalLMNextN._resolve_nextn_quant_config(
            object(), SimpleNamespace(), quant_config
        )
        self.assertIs(resolved, quant_config)


if __name__ == "__main__":
    import unittest

    unittest.main()
