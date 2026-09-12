#!/usr/bin/env python3
"""Reproducer from sgl-project/sglang#35564, with consistent call-index comparison."""
import json
import warnings

warnings.filterwarnings("ignore")

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser


def tool(name):
    return Tool(
        type="function",
        function=Function(name=name, parameters={"type": "object", "properties": {}}),
    )


TOOLS = [tool("get_weather"), tool("f")]
CASES = {
    "cohere_command4": '<|START_ACTION|>[{"tool_name": "f", "parameters": {}}]<|END_ACTION|>',
    "gemma4": "<|tool_call>call:f{}<tool_call|>",
    "glm": "<tool_call>get_weather\n<arg_key>get_weather</arg_key>\n<arg_value>123</arg_value>\n</tool_call>",
    "glm45": "<tool_call>get_weather\n<arg_key>get_weather</arg_key>\n<arg_value>123</arg_value>\n</tool_call>",
    "glm47": "<tool_call>get_weather<arg_key>get_weather</arg_key><arg_value>123</arg_value></tool_call>",
    "minimax-m2": '<minimax:tool_call><invoke name="get_weather"></invoke></minimax:tool_call>',
    "mistral": '[TOOL_CALLS] [{"name": "get_weather", "arguments": {}}, {"name": "get_weather", "arguments": {}}]',
    "step3": "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
    '<steptml:invoke name="get_weather"></steptml:invoke>'
    "<｜tool_call_end｜><｜tool_call_begin｜>function<｜tool_sep｜>"
    '<steptml:invoke name="get_weather">'
    '<steptml:parameter name="get_weather">hello</steptml:parameter>'
    "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>",
}


def stream(name, chunks):
    detector = FunctionCallParser.ToolCallParserEnum[name]()
    calls = {}
    for chunk in list(chunks) + ["", ""]:
        result = detector.parse_streaming_increment(chunk, TOOLS)
        for call in result.calls:
            entry = calls.setdefault(call.tool_index, ["", ""])
            if call.name:
                entry[0] = call.name
            if call.parameters:
                entry[1] += call.parameters
    return calls


def json_equal(left, right):
    try:
        return json.loads(left or "null") == json.loads(right or "null")
    except Exception:
        return left == right


bad = 0
for name, text in CASES.items():
    detector = FunctionCallParser.ToolCallParserEnum[name]()
    final = {
        call.tool_index: [call.name, call.parameters]
        for call in detector.detect_and_parse(text, TOOLS).calls
    }
    streamed = stream(name, list(text))
    same = len(streamed) == len(final) and all(
        streamed.get(index, ["", ""])[0] == expected[0]
        and json_equal(streamed.get(index, ["", ""])[1], expected[1])
        for index, expected in final.items()
    )
    print(f"{'OK ' if same else 'BUG'} {name:16s} final={final}")
    if not same:
        print(f"    streamed={streamed}")
        bad += 1

raise SystemExit(1 if bad else 0)
