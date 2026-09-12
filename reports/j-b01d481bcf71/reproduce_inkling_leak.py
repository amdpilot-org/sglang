#!/usr/bin/env python3
"""Issue #33181 parser-path reproduction against the checked-out source."""

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import InklingDetector

CASES = {
    "missing opener": 'get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}<|end_message|>',
    "opener present": '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}<|end_message|>',
    "thinking first": '<|content_thinking|>Checking the forecast.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{}}<|end_message|>',
}
TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="get_weather",
            description="Get weather.",
            parameters={"type": "object", "properties": {}},
        ),
    )
]


def summarize(mode: str, source: str):
    reasoning_detector = InklingDetector()
    if mode == "one-shot":
        parsed = reasoning_detector.detect_and_parse(source)
        normal_text = parsed.normal_text
        reasoning_text = parsed.reasoning_text
    else:
        normal_parts = []
        reasoning_parts = []
        for start in range(0, len(source), 7):
            parsed = reasoning_detector.parse_streaming_increment(
                source[start : start + 7]
            )
            normal_parts.append(parsed.normal_text)
            reasoning_parts.append(parsed.reasoning_text)
        parsed = reasoning_detector.finish()
        normal_parts.append(parsed.normal_text)
        reasoning_parts.append(parsed.reasoning_text)
        normal_text = "".join(normal_parts)
        reasoning_text = "".join(reasoning_parts)

    visible_text, tool_calls = FunctionCallParser(TOOLS, "inkling").parse_non_stream(
        normal_text
    )
    calls = [f"{call.name}({call.parameters})" for call in tool_calls]
    return reasoning_text, visible_text, calls


print("case | mode | reasoning | visible | tool calls")
for case, source in CASES.items():
    for mode in ("one-shot", "streamed"):
        reasoning, visible, calls = summarize(mode, source)
        print(f"{case} | {mode} | {reasoning!r} | {visible!r} | {calls!r}")
