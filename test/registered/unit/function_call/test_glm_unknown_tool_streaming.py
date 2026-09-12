import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")


TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="known", parameters={"type": "object", "properties": {}}
        ),
    )
]


def _stream(detector, text):
    return [
        call
        for char in text
        for call in detector.parse_streaming_increment(char, TOOLS).calls
    ]


@pytest.mark.parametrize(
    ("detector_cls", "unknown_call"),
    [
        (Glm4MoeDetector, "<tool_call>unknown\n</tool_call>"),
        (Glm47MoeDetector, "<tool_call>unknown</tool_call>"),
    ],
)
@pytest.mark.parametrize("forward_unknown", [False, True])
def test_streaming_unknown_tool_policy(detector_cls, unknown_call, forward_unknown):
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward_unknown):
        calls = _stream(detector_cls(), unknown_call)

    if forward_unknown:
        assert [(call.tool_index, call.name, call.parameters) for call in calls] == [
            (0, "unknown", ""),
            (0, None, "{}"),
        ]
    else:
        assert calls == []


@pytest.mark.parametrize("single_chunk", [False, True])
@pytest.mark.parametrize(
    ("detector_cls", "text"),
    [
        (
            Glm4MoeDetector,
            "<tool_call>unknown\n</tool_call><tool_call>known\n</tool_call>",
        ),
        (
            Glm47MoeDetector,
            "<tool_call>unknown</tool_call><tool_call>known</tool_call>",
        ),
    ],
)
def test_dropped_unknown_tool_does_not_corrupt_following_call(
    detector_cls, text, single_chunk
):
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        if single_chunk:
            calls = detector_cls().parse_streaming_increment(text, TOOLS).calls
        else:
            calls = _stream(detector_cls(), text)

    assert [(call.tool_index, call.name, call.parameters) for call in calls] == [
        (0, "known", ""),
        (0, None, "{}"),
    ]
