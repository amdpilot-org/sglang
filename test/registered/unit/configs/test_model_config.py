"""Unit tests for hybrid attention model configuration."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from transformers import DeepseekV4Config

from sglang.srt.configs.model_config import (
    ModelConfig,
    _normalize_deepseek_v4_config,
    get_hybrid_layer_ids,
    get_num_indexer_layers,
    is_embedding_gemma,
    is_multimodal_model,
    resolve_spec_hidden_size,
)
from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.utils.hf_transformers import get_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDeepseekV4Transformers457Config(CustomTestCase):
    def test_generated_config_loads_and_normalizes_runtime_fields(self):
        generated = DeepseekV4Config(
            architectures=["DeepseekV4ForCausalLM"], num_hidden_layers=4
        )
        with TemporaryDirectory() as directory:
            generated.save_pretrained(Path(directory))
            config = get_config(directory, trust_remote_code=False)

        ratios, rope_scaling = _normalize_deepseek_v4_config(config)

        self.assertEqual(ratios, [128, 128, 128, 4])
        self.assertEqual(config.compress_ratios, ratios)
        self.assertEqual(get_num_indexer_layers(config), 1)
        self.assertEqual(rope_scaling, config.rope_parameters["compress"])
        self.assertEqual(
            config.rope_theta, config.rope_parameters["main"]["rope_theta"]
        )
        self.assertEqual(
            config.compress_rope_theta,
            config.rope_parameters["compress"]["rope_theta"],
        )

    def test_dictionary_rates_follow_layer_types_not_dictionary_order(self):
        config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            num_hidden_layers=4,
            layer_types=["hca", "csa", "hca", "csa"],
            compress_rates={"csa": 4, "hca": 128},
            rope_parameters={
                "main": {"rope_type": "default", "rope_theta": 10_000},
                "compress": {"rope_type": "yarn", "rope_theta": 160_000, "factor": 16},
            },
        )

        ratios, rope_scaling = _normalize_deepseek_v4_config(config)

        self.assertEqual(ratios, [128, 4, 128, 4])
        self.assertEqual(get_num_indexer_layers(config), 2)
        self.assertEqual(rope_scaling, config.rope_parameters["compress"])
        self.assertEqual(config.rope_theta, 10_000)
        self.assertEqual(config.compress_rope_theta, 160_000)

    def test_legacy_fields_remain_supported(self):
        config = SimpleNamespace(
            num_hidden_layers=2,
            compress_ratios=[4, 0],
            rope_scaling={"type": "yarn", "factor": 4},
        )

        ratios, rope_scaling = _normalize_deepseek_v4_config(config)

        self.assertEqual(ratios, [4, 0])
        self.assertEqual(rope_scaling, {"type": "yarn", "factor": 4})

    def test_conflicting_or_incomplete_fields_are_rejected(self):
        conflicting = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["hca", "csa"],
            compress_rates={"hca": 128, "csa": 4},
            compress_ratios=[4, 128],
        )
        with self.assertRaisesRegex(ValueError, "conflicting compress_rates"):
            _normalize_deepseek_v4_config(conflicting)

        missing_layer_types = SimpleNamespace(
            num_hidden_layers=2, compress_rates={"hca": 128}
        )
        with self.assertRaisesRegex(ValueError, "requires layer_types"):
            _normalize_deepseek_v4_config(missing_layer_types)

        wrong_length = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["hca"],
            compress_rates={"hca": 128},
        )
        with self.assertRaisesRegex(ValueError, "one entry per layer"):
            _normalize_deepseek_v4_config(wrong_length)


class TestHybridLayerIds(CustomTestCase):
    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )


class TestEmbeddingGemmaConfig(CustomTestCase):
    def test_detects_bidirectional_gemma3_text_config(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=True
        )
        self.assertTrue(is_embedding_gemma(config))

    def test_does_not_misclassify_causal_gemma3(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=False
        )
        self.assertFalse(is_embedding_gemma(config))


class TestDraftModelConfig(CustomTestCase):
    def test_nemotron_h_omni_is_multimodal(self):
        self.assertTrue(is_multimodal_model(["NemotronH_Omni_Reasoning_V3"]))

    def test_qwen35_mtp_depth_is_synced_to_text_config(self):
        config = object.__new__(ModelConfig)
        config.is_draft_model = True
        config.speculative_algorithm = "EAGLE"
        config.hf_config = SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"]
        )
        config.hf_text_config = SimpleNamespace()

        config._config_draft_model()

        self.assertEqual(config.hf_config.architectures, ["Qwen3_5ForCausalLMMTP"])
        self.assertEqual(config.hf_config.num_nextn_predict_layers, 1)
        self.assertEqual(config.hf_text_config.num_nextn_predict_layers, 1)

    def test_nemotron_h_omni_mtp_uses_language_model_config(self):
        config = object.__new__(ModelConfig)
        config.is_draft_model = True
        config.speculative_algorithm = "EAGLE"
        config.hf_config = SimpleNamespace(
            architectures=["NemotronH_Omni_Reasoning_V3"]
        )
        config.hf_text_config = SimpleNamespace(architectures=["NemotronHForCausalLM"])

        config._config_draft_model()

        self.assertIs(config.hf_config, config.hf_text_config)
        self.assertEqual(config.hf_config.architectures, ["NemotronHForCausalLMMTP"])
        self.assertEqual(config.hf_config.num_nextn_predict_layers, 1)

    def test_qwen4_exp_spec_hidden_size_keeps_hc_width(self):
        """Qwen4-Exp's MTP draft consumes the hc-flattened target stream,
        so spec_hidden_size must stay hidden_size * hc_mult; hy_v4 collapses first."""
        hidden_size, hc_mult = 2560, 4
        self.assertEqual(Qwen4ExpTextConfig(hc_count=hc_mult).hc_mult, hc_mult)
        for arch in ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLMMTP"):
            hf_config = SimpleNamespace(architectures=[arch])
            self.assertEqual(
                resolve_spec_hidden_size(
                    hf_config=hf_config, hidden_size=hidden_size, hc_mult=hc_mult
                ),
                (hidden_size * hc_mult, hidden_size * hc_mult),
            )
        hy_v4 = SimpleNamespace(architectures=["HYV4ForCausalLM"])
        self.assertEqual(
            resolve_spec_hidden_size(
                hf_config=hy_v4, hidden_size=hidden_size, hc_mult=hc_mult
            ),
            (hidden_size, None),
        )


if __name__ == "__main__":
    unittest.main()
