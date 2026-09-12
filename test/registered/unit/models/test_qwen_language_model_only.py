from types import SimpleNamespace

from sglang.srt.arg_groups.model_hook import handle_language_model_only
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    is_qwen_visual_weight,
)
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwenLanguageModelOnly(CustomTestCase):
    SUPPORTED_ARCHITECTURES = (
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    )

    def test_qwen_vl_architectures_accept_language_model_only(self):
        for architecture in self.SUPPORTED_ARCHITECTURES:
            with self.subTest(architecture=architecture):
                args = ServerArgs(model_path="dummy", language_model_only=True)
                args._model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=[architecture])
                )

                handle_language_model_only(args)

    def test_unrelated_architecture_remains_rejected(self):
        args = ServerArgs(model_path="dummy", language_model_only=True)
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3VLForSequenceClassification"]
            )
        )

        with self.assertRaisesRegex(ValueError, "does not support"):
            handle_language_model_only(args)

    def test_visual_weight_filter_uses_exact_tower_prefixes(self):
        for name in (
            "visual.patch_embed.proj.weight",
            "model.visual.blocks.0.attn.qkv.weight",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_qwen_visual_weight(name))

        for name in (
            "model.language_model.layers.0.visual_gate.weight",
            "model.visualizer.weight",
            "prefix.model.visual.blocks.0.weight",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_qwen_visual_weight(name))

    def test_language_only_loaders_do_not_touch_visual_tensors(self):
        class LanguageOnlyModel:
            language_model_only = True
            config = SimpleNamespace(num_experts=1)
            enable_shared_expert_fusion = False
            num_fused_shared_experts = 0

            @staticmethod
            def named_parameters(*args, **kwargs):
                return ()

        weights = [
            ("model.visual.blocks.0.attn.qkv.weight", object()),
            ("visual.patch_embed.proj.weight", object()),
        ]
        for model_class in (
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        ):
            with self.subTest(model_class=model_class.__name__):
                model_class.load_weights(LanguageOnlyModel(), iter(weights))


if __name__ == "__main__":
    import unittest

    unittest.main()
