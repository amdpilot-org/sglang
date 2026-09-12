"""Unit tests for DeepSeek DSML streaming — no server, no model loading."""

import json
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


class TestDeepSeekV4Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        self.assertNotIn(DSML, normal)

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        """A partial non-string parameter must not escape as MalformedJSON."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_parse_error_neither_swallows_nor_duplicates(self):
        """An unexpected parse error must not empty the turn, and the dropped
        buffer must not come back on the next delta."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(_weather_call(), self.tools)
            self.assertEqual(detector._buffer, "")
            second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertIn("get_weather", first.normal_text)
        self.assertNotIn("get_weather", second.normal_text)
        # No half-formed call: the failure can land between a tool's name and its
        # arguments, so an argument-less named call must not reach the client.
        self.assertEqual(first.calls, [])


class TestDeepSeekWrappedArguments(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="bash",
                    description="Run a shell command",
                    parameters={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="literal_input",
                    description="A tool with a real input parameter",
                    parameters={
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="read",
                    description="Read a file",
                    parameters={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="literal_arguments",
                    description="A tool with a real arguments parameter",
                    parameters={
                        "type": "object",
                        "properties": {"arguments": {"type": "string"}},
                        "required": ["arguments"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="pair",
                    description="A two-parameter tool",
                    parameters={
                        "type": "object",
                        "properties": {
                            "left": {"type": "string"},
                            "right": {"type": "string"},
                        },
                        "required": ["left", "right"],
                    },
                ),
            ),
        ]

    @staticmethod
    def _accumulate(detector, chunks, tools):
        calls = {}
        per_chunk_parameters = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            emitted = "".join(call.parameters for call in result.calls)
            per_chunk_parameters.append(emitted)
            for call in result.calls:
                entry = calls.setdefault(
                    call.tool_index, {"name": "", "arguments": ""}
                )
                if call.name:
                    entry["name"] = call.name
                entry["arguments"] += call.parameters
        return calls, per_chunk_parameters

    def test_one_shot_original_payload_forms_for_v32_and_v4(self):
        cases = (
            (
                '<｜DSML｜invoke name="read">'
                '{"arguments":{"path":"/some/file.py"}}</｜DSML｜invoke>',
                {"path": "/some/file.py"},
            ),
            (
                '<｜DSML｜invoke name="bash">{"input":"git status"}</｜DSML｜invoke>',
                {"command": "git status"},
            ),
            (
                '<｜DSML｜invoke name="read"><｜DSML｜parameter name="arguments" string="true">'
                '{"path":"/some/file.py"}</｜DSML｜parameter></｜DSML｜invoke>',
                {"path": "/some/file.py"},
            ),
        )
        for detector_type, outer_tag in (
            (DeepSeekV32Detector, "function_calls"),
            (DeepSeekV4Detector, "tool_calls"),
        ):
            for invoke, expected in cases:
                text = f"<｜DSML｜{outer_tag}>{invoke}</｜DSML｜{outer_tag}>"
                result = detector_type().detect_and_parse(text, self.tools)
                self.assertEqual(len(result.calls), 1)
                self.assertEqual(json.loads(result.calls[0].parameters), expected)

    def test_streaming_holds_ambiguous_shape_until_complete(self):
        invoke = (
            '<｜DSML｜invoke name="bash">\n'
            '{"arguments":"git status"}\n</｜DSML｜invoke>'
        )
        for detector_type, outer_tag in (
            (DeepSeekV32Detector, "function_calls"),
            (DeepSeekV4Detector, "tool_calls"),
        ):
            text = f"<｜DSML｜{outer_tag}>\n{invoke}\n</｜DSML｜{outer_tag}>"
            split = text.index("git status") + 3
            calls, emitted = self._accumulate(
                detector_type(), [text[:split], text[split:]], self.tools
            )

            self.assertEqual(emitted[0], "")
            self.assertEqual(calls[0]["name"], "bash")
            self.assertEqual(
                json.loads(calls[0]["arguments"]), {"command": "git status"}
            )

    def test_streaming_xml_wrapper_across_every_boundary(self):
        text = _wrapped(
            '<｜DSML｜invoke name="read">'
            '<｜DSML｜parameter name="input" string="true">'
            '{"path":"/tmp/a"}</｜DSML｜parameter></｜DSML｜invoke>'
        )
        for split in range(1, len(text)):
            calls, _ = self._accumulate(
                DeepSeekV4Detector(), [text[:split], text[split:]], self.tools
            )
            self.assertEqual(json.loads(calls[0]["arguments"]), {"path": "/tmp/a"})

    def test_real_parameter_names_and_unsupported_scalar_are_preserved(self):
        literal = _wrapped(
            '<｜DSML｜invoke name="literal_arguments">'
            '{"arguments":"--verbose"}</｜DSML｜invoke>'
        )
        result = DeepSeekV4Detector().detect_and_parse(literal, self.tools)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"arguments": "--verbose"}
        )

        literal = _wrapped(
            '<｜DSML｜invoke name="literal_input">'
            '{"input":"payload"}</｜DSML｜invoke>'
        )
        result = DeepSeekV4Detector().detect_and_parse(literal, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"input": "payload"})

        pair = _wrapped(
            '<｜DSML｜invoke name="pair">{"input":"value"}</｜DSML｜invoke>'
        )
        result = DeepSeekV4Detector().detect_and_parse(pair, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"input": "value"})


if __name__ == "__main__":
    import unittest

    unittest.main()
