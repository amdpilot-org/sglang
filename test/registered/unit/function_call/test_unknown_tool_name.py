import json
import logging

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
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


STREAMING_TOOLS = [
    Tool(function=Function(name="get_weather", parameters={})),
    Tool(function=Function(name="get_time", parameters={})),
]
WEATHER_CALL = '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
TIME_CALL = '{"name": "get_time", "arguments": {"tz": "JST"}}'
UNKNOWN_CALL = '{"name": "rm_rf", "arguments": {"path": "/"}}'


def _collect_streamed_calls(detector, chunks):
    calls = {}
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, STREAMING_TOOLS)
        for call in result.calls or []:
            slot = calls.setdefault(call.tool_index, {"name": None, "arguments": ""})
            if call.name:
                slot["name"] = call.name
            if call.parameters:
                slot["arguments"] += call.parameters
    return calls


@pytest.mark.parametrize(
    ("detector", "wire"),
    [
        (
            JsonArrayParser(),
            "["
            + ",".join(
                call.replace('"arguments"', '"parameters"')
                for call in (UNKNOWN_CALL, WEATHER_CALL, TIME_CALL)
            )
            + "]",
        ),
        (
            Qwen25Detector(),
            "".join(
                f"<tool_call>\n{call}\n</tool_call>\n"
                for call in (WEATHER_CALL, UNKNOWN_CALL, TIME_CALL)
            ),
        ),
    ],
)
def test_streaming_unknown_tool_preserves_parallel_calls(detector, wire, caplog):
    with caplog.at_level(
        logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
    ):
        calls = _collect_streamed_calls(detector, wire)

    assert calls == {
        0: {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
        1: {"name": "get_time", "arguments": '{"tz": "JST"}'},
    }
    assert (
        caplog.messages.count("Model attempted to call undefined function: rm_rf") == 1
    )


def test_streaming_unknown_tool_last_preserves_completed_state():
    detector = JsonArrayParser()
    wire = (
        "["
        + ",".join(
            call.replace('"arguments"', '"parameters"')
            for call in (WEATHER_CALL, UNKNOWN_CALL)
        )
        + "]"
    )

    calls = _collect_streamed_calls(detector, [wire, "", ""])

    assert calls == {0: {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}
    assert detector.streamed_args_for_tool == ['{"city": "Tokyo"}']
    assert detector.prev_tool_call_arr[0]["arguments"] == {"city": "Tokyo"}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
