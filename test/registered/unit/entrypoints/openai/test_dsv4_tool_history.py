"""Regression tests for DeepSeek V4 assistant tool-call history encoding."""

import json
import unittest

from sglang.srt.entrypoints.openai import encoding_dsv4
from sglang.srt.entrypoints.openai.serving_chat import (
    normalize_assistant_tool_call_arguments,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestDeepSeekV4ToolHistory(CustomTestCase):
    @staticmethod
    def _history(arguments):
        return [
            {"role": "user", "content": "Read both files."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I will read the first file.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {"role": "user", "content": "Now read the second file."},
        ]

    def test_normalized_openai_arguments_do_not_gain_arguments_parameter(self):
        messages = self._history('{"filePath":"/etc/hosts"}')

        # serving_chat normalizes OpenAI's JSON string before the DSV4 encoder.
        normalize_assistant_tool_call_arguments(messages[1])
        prompt = encoding_dsv4.encode_messages(messages, thinking_mode="thinking")

        self.assertIn(
            '<｜DSML｜parameter name="filePath" string="true">/etc/hosts'
            "</｜DSML｜parameter>",
            prompt,
        )
        self.assertNotIn('<｜DSML｜parameter name="arguments"', prompt)

    def test_json_string_and_normalized_dict_encode_identically(self):
        arguments = {
            "filePath": "/etc/hosts",
            "offset": 0,
            "options": {"lines": [1, 3], "required": False},
        }
        from_string = encoding_dsv4.encode_messages(
            self._history(json.dumps(arguments)), thinking_mode="chat"
        )
        from_dict = encoding_dsv4.encode_messages(
            self._history(arguments), thinking_mode="chat"
        )

        self.assertEqual(from_string, from_dict)
        self.assertIn('<｜DSML｜parameter name="offset" string="false">0', from_dict)
        self.assertIn(
            '<｜DSML｜parameter name="options" string="false">'
            '{"lines": [1, 3], "required": false}',
            from_dict,
        )

    def test_non_object_arguments_are_rejected(self):
        for arguments in ('["/etc/hosts"]', ["/etc/hosts"]):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                ValueError, "must be a JSON object"
            ):
                encoding_dsv4.encode_messages(
                    self._history(arguments), thinking_mode="chat"
                )


if __name__ == "__main__":
    unittest.main()
