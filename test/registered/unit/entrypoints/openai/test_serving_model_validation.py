import json
import unittest
from unittest.mock import Mock

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class MinimalServing(OpenAIServingBase):
    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        raise AssertionError("unknown models must be rejected before conversion")


class ServedModelValidationTest(unittest.TestCase):
    def setUp(self):
        manager = Mock()
        manager.served_model_name = "served-model"
        manager.server_args.enable_lora = False
        manager.server_args.tokenizer_metrics_allowed_custom_labels = None
        manager.lora_registry.get_all_adapters.return_value = {}
        self.manager = manager
        self.serving = MinimalServing(manager)

    @staticmethod
    def error(response):
        return json.loads(response.body)["error"]

    def test_unknown_model_returns_openai_404(self):
        response = self.serving.validate_served_model(
            CompletionRequest(model="missing-model", prompt="hello")
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            self.error(response),
            {
                "message": "The model 'missing-model' does not exist",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            },
        )

    def test_served_and_omitted_models_are_accepted(self):
        self.assertIsNone(
            self.serving.validate_served_model(
                CompletionRequest(model="served-model", prompt="hello")
            )
        )
        self.assertIsNone(
            self.serving.validate_served_model(CompletionRequest(prompt="hello"))
        )

    def test_lora_requires_matching_base_and_registered_adapter(self):
        self.manager.server_args.enable_lora = True
        self.manager.lora_registry.get_all_adapters.return_value = {"loaded": Mock()}

        self.assertIsNone(
            self.serving.validate_served_model(
                CompletionRequest(model="served-model:loaded", prompt="hello")
            )
        )
        for model in ("served-model:missing", "other-model:loaded"):
            with self.subTest(model=model):
                response = self.serving.validate_served_model(
                    CompletionRequest(model=model, prompt="hello")
                )
                self.assertEqual(response.status_code, 404)

    def test_explicit_default_is_not_treated_as_omitted(self):
        response = self.serving.validate_served_model(
            CompletionRequest(model="default", prompt="hello")
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
