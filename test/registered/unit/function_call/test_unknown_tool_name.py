import json
import logging

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")


class DummyDetector(BaseFormatDetector):
    def has_tool_call(self, text: str) -> bool:
        return True

    def detect_and_parse(self, text: str, tools):
        action = json.loads(text)
        return StreamingParseResult(
            normal_text="", calls=self.parse_base_json(action, tools)
        )

    def structure_info(self):
        pass


def test_unknown_tool_name_dropped_default(caplog):
    """Test that unknown tools are dropped by default (legacy behavior)."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        tools = [
            Tool(
                function=Function(
                    name="get_weather", parameters={"type": "object", "properties": {}}
                )
            )
        ]
        detector = DummyDetector()
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            result = detector.detect_and_parse(
                '{"name":"unknown_tool","parameters":{"city":"Paris"}}', tools
            )
        assert any(
            "Model attempted to call undefined function: unknown_tool" in m
            for m in caplog.messages
        )
        assert len(result.calls) == 0  # dropped in default mode


def test_unknown_tool_name_forwarded(caplog):
    """Test that unknown tools are forwarded when env var is True."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        tools = [
            Tool(
                function=Function(
                    name="get_weather", parameters={"type": "object", "properties": {}}
                )
            )
        ]
        detector = DummyDetector()
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            result = detector.detect_and_parse(
                '{"name":"unknown_tool","parameters":{"city":"Paris"}}', tools
            )
        assert any(
            "Model attempted to call undefined function: unknown_tool" in m
            for m in caplog.messages
        )
        assert len(result.calls) == 1
        assert result.calls[0].name == "unknown_tool"
        assert result.calls[0].tool_index == -1
        assert json.loads(result.calls[0].parameters)["city"] == "Paris"


def test_non_object_entry_does_not_discard_valid_qwen25_call(caplog):
    tools = [
        Tool(
            function=Function(
                name="get_weather", parameters={"type": "object", "properties": {}}
            )
        )
    ]
    parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")

    with caplog.at_level(
        logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
    ):
        normal_text, calls = parser.parse_non_stream(
            '<tool_call>\n[{"name": "get_weather", "arguments": '
            '{"city": "Paris"}}, "junk"]\n</tool_call>'
        )

    assert normal_text == ""
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].parameters) == {"city": "Paris"}
    assert any("non-object payload: 'junk'" in m for m in caplog.messages)


@pytest.mark.parametrize("invalid_entry", [1, "junk", None, []])
def test_non_object_entries_are_skipped_independently(invalid_entry):
    tools = [
        Tool(
            function=Function(
                name="get_weather", parameters={"type": "object", "properties": {}}
            )
        )
    ]
    detector = DummyDetector()
    valid_call = {"name": "get_weather", "arguments": {"city": "Paris"}}

    result = detector.parse_base_json([invalid_entry, valid_call], tools)

    assert len(result) == 1
    assert result[0].name == "get_weather"
    assert json.loads(result[0].parameters) == {"city": "Paris"}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
