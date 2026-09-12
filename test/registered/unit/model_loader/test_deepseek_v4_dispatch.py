"""Regression coverage for DeepSeek-V4 config-to-model dispatch."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.model_loader.utils import get_model_architecture
from sglang.srt.utils.hf_transformers import get_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV4Dispatch(unittest.TestCase):
    def test_alias_config_uses_native_model(self):
        config = {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "torch_dtype": "bfloat16",
        }
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(json.dumps(config))
            hf_config = get_config(model_dir, trust_remote_code=True)

        self.assertEqual(type(hf_config).__name__, "_DeepseekV4ConfigAlias")
        self.assertEqual(hf_config.model_type, "deepseek_v4")
        model_config = SimpleNamespace(
            hf_config=hf_config,
            is_embedding_gemma=False,
            quantization=None,
            model_impl=ModelImpl.AUTO,
        )

        # The alias is intentionally only a config parser.  It is not registered
        # with HF AutoModelForCausalLM, so auto dispatch must select SGLang's
        # native implementation before the Transformers fallback is considered.
        with patch(
            "sglang.srt.model_loader.utils.resolve_transformers_arch",
            side_effect=AssertionError("DeepSeek-V4 unexpectedly used HF fallback"),
        ):
            model_cls, architecture = get_model_architecture(model_config)

        self.assertEqual(architecture, "DeepseekV4ForCausalLM")
        self.assertEqual(model_cls.__module__, "sglang.srt.models.deepseek_v4")

    def test_unknown_architecture_still_uses_transformers_fallback(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ExternalForCausalLM"]),
            is_embedding_gemma=False,
            quantization=None,
            model_impl=ModelImpl.AUTO,
        )

        with patch(
            "sglang.srt.model_loader.utils.resolve_transformers_arch",
            return_value=["TransformersForCausalLM"],
        ) as fallback:
            _, architecture = get_model_architecture(model_config)

        fallback.assert_called_once_with(model_config, ["ExternalForCausalLM"])
        self.assertEqual(architecture, "TransformersForCausalLM")


if __name__ == "__main__":
    unittest.main()
