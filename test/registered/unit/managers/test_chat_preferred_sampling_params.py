import unittest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.managers.tokenizer_manager import (
    merge_preferred_sampling_params,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestChatPreferredSamplingParams(unittest.TestCase):
    def test_all_client_sampling_fields_are_tracked(self):
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=9,
            min_tokens=2,
            stop=["done"],
            stop_token_ids=[3],
            stop_regex="end",
            temperature=0.6,
            top_p=0.7,
            top_k=8,
            min_p=0.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            repetition_penalty=1.1,
            regex="a+",
            ebnf="root ::= 'a'",
            n=2,
            no_stop_trim=True,
            ignore_eos=True,
            skip_special_tokens=False,
            logit_bias={"1": 2.0},
            custom_params={"x": 1},
            seed=4,
            response_format={"type": "json_object"},
            chat_template_kwargs={"spaces_between_special_tokens": False},
        )

        self.assertEqual(
            set(request.get_explicit_sampling_keys()),
            {
                "max_new_tokens",
                "min_new_tokens",
                "stop",
                "stop_token_ids",
                "stop_regex",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
                "regex",
                "ebnf",
                "n",
                "no_stop_trim",
                "ignore_eos",
                "skip_special_tokens",
                "logit_bias",
                "custom_params",
                "sampling_seed",
                "json_schema",
                "structural_tag",
                "spaces_between_special_tokens",
            },
        )

    def test_unset_fields_prefer_server_values_over_generation_config(self):
        request = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.3,
            "top_k": 3,
            "min_p": 0.04,
            "repetition_penalty": 1.2,
        }
        preferred = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.05,
            "presence_penalty": 1.5,
            "frequency_penalty": 1.25,
            "repetition_penalty": 1.1,
            "n": 2,
            "no_stop_trim": True,
            "ignore_eos": True,
            "skip_special_tokens": False,
            "sampling_seed": 17,
        }

        converted = request.to_sampling_params([], generation_config)
        effective = merge_preferred_sampling_params(
            converted, preferred, request.get_explicit_sampling_keys()
        )

        for key, value in preferred.items():
            self.assertEqual(effective[key], value, key)

    def test_explicit_client_values_override_preferences_including_defaults(self):
        explicit = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "n": 1,
            "no_stop_trim": False,
            "ignore_eos": False,
            "skip_special_tokens": True,
            "seed": 0,
        }
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            **explicit,
        )
        preferred = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.05,
            "presence_penalty": 1.5,
            "frequency_penalty": 1.25,
            "repetition_penalty": 1.1,
            "n": 2,
            "no_stop_trim": True,
            "ignore_eos": True,
            "skip_special_tokens": False,
            "sampling_seed": 17,
        }

        converted = request.to_sampling_params([], {})
        effective = merge_preferred_sampling_params(
            converted, preferred, request.get_explicit_sampling_keys()
        )

        expected = {**explicit, "sampling_seed": explicit["seed"]}
        expected.pop("seed")
        for key, value in expected.items():
            self.assertEqual(effective[key], value, key)

    def test_generation_config_survives_for_keys_without_preferences(self):
        request = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        converted = request.to_sampling_params([], {"temperature": 0.2, "top_p": 0.3})

        effective = merge_preferred_sampling_params(
            converted,
            {"presence_penalty": 1.5},
            request.get_explicit_sampling_keys(),
        )

        self.assertEqual(effective["temperature"], 0.2)
        self.assertEqual(effective["top_p"], 0.3)
        self.assertEqual(effective["presence_penalty"], 1.5)

    def test_legacy_native_request_keeps_request_over_preference_behavior(self):
        self.assertEqual(
            merge_preferred_sampling_params(
                {"temperature": 0.4}, {"temperature": 0.7}, None
            )["temperature"],
            0.4,
        )


if __name__ == "__main__":
    unittest.main()
