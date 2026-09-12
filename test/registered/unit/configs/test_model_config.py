"""Unit tests for hybrid attention model configuration."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    ModelConfig,
    _get_deepseek_v4_config_value,
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
    @staticmethod
    def _write_config(path: Path, **overrides):
        import json

        config = {
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
            "compress_rates": [0, 4, 128, 0],
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4096,
                "rope_theta": 10000,
            },
        }
        config.update(overrides)
        (path / "config.json").write_text(json.dumps(config))

    def test_new_schema_loads_and_exposes_legacy_runtime_aliases(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self._write_config(path)
            config = get_config(str(path), trust_remote_code=False)

        self.assertEqual(config.compress_rates, [0, 4, 128, 0])
        self.assertEqual(config.compress_ratios, [0, 4, 128, 0])
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertEqual(config.rope_scaling["factor"], 2.0)
        self.assertEqual(get_num_indexer_layers(config), 1)

    def test_legacy_schema_remains_supported(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self._write_config(
                path,
                compress_rates=None,
                compress_ratios=[4, 0],
                rope_parameters=None,
                rope_scaling={"rope_type": "linear", "factor": 4.0},
            )
            config = get_config(str(path), trust_remote_code=False)

        self.assertEqual(config.compress_rates, [4, 0])
        self.assertEqual(config.compress_ratios, [4, 0])
        self.assertEqual(config.rope_parameters["factor"], 4.0)

    def test_equal_old_and_new_names_are_accepted(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self._write_config(
                path,
                compress_ratios=[0, 4, 128, 0],
                rope_scaling={
                    "rope_type": "yarn",
                    "factor": 2.0,
                    "original_max_position_embeddings": 4096,
                    "rope_theta": 10000,
                },
            )
            config = get_config(str(path), trust_remote_code=False)

        self.assertEqual(config.compress_ratios, config.compress_rates)
        self.assertEqual(config.rope_scaling, config.rope_parameters)

    def test_conflicting_aliases_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self._write_config(path, compress_ratios=[4, 4, 4, 4])
            with self.assertRaisesRegex(ValueError, "conflicting compress_rates"):
                get_config(str(path), trust_remote_code=False)

    def test_remote_config_aliases_are_normalized_and_validated(self):
        remote_config = SimpleNamespace(
            compress_rates=[0, 4], rope_parameters={"factor": 2.0}
        )
        self.assertEqual(
            _get_deepseek_v4_config_value(
                remote_config, "compress_rates", "compress_ratios"
            ),
            [0, 4],
        )
        self.assertEqual(
            _get_deepseek_v4_config_value(
                remote_config, "rope_parameters", "rope_scaling"
            ),
            {"factor": 2.0},
        )

        conflicting = SimpleNamespace(compress_rates=[0, 4], compress_ratios=[4, 0])
        with self.assertRaisesRegex(ValueError, "conflicting compress_rates"):
            _get_deepseek_v4_config_value(
                conflicting, "compress_rates", "compress_ratios"
            )


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
