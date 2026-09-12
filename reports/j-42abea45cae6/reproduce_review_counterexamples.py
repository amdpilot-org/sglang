"""Reproduce independent-review counterexamples against candidate PR 1754."""

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.mistral_detector import MistralDetector


TOOLS = [
    Tool(function=Function(name="get_weather", parameters={})),
    Tool(function=Function(name="get_time", parameters={})),
]
WEATHER = '{"name":"get_weather","arguments":{"city":"Tokyo"}}'
UNKNOWN = '{"name":"rm_rf","arguments":{"path":"/"}}'
TIME = '{"name":"get_time","arguments":{"tz":"JST"}}'


def collect(detector, chunks):
    calls = {}
    normal_text = ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, TOOLS)
        normal_text += result.normal_text or ""
        for call in result.calls or []:
            slot = calls.setdefault(call.tool_index, {"name": None, "arguments": ""})
            if call.name:
                slot["name"] = call.name
            if call.parameters:
                slot["arguments"] += call.parameters
    result = detector.finish(TOOLS)
    normal_text += result.normal_text or ""
    for call in result.calls or []:
        slot = calls.setdefault(call.tool_index, {"name": None, "arguments": ""})
        if call.name:
            slot["name"] = call.name
        if call.parameters:
            slot["arguments"] += call.parameters
    return calls, normal_text, detector._buffer


for separator in (", ", ",", ",\n", ",\t"):
    wire = "[TOOL_CALLS] [" + separator.join((WEATHER, UNKNOWN, TIME)) + "]"
    print("mistral", repr(separator), collect(MistralDetector(), wire))

for chunking in ("whole", "coarse"):
    valid = "".join(f"<tool_call>{call}</tool_call>" for call in (WEATHER, TIME))
    mixed = "".join(
        f"<tool_call>{call}</tool_call>" for call in (WEATHER, UNKNOWN, TIME)
    )
    chunks = lambda wire: [wire] if chunking == "whole" else [wire[i : i + 47] for i in range(0, len(wire), 47)]
    valid_result = collect(HermesDetector(), chunks(valid))
    mixed_result = collect(HermesDetector(), chunks(mixed))
    print("hermes", chunking, "valid", valid_result)
    print("hermes", chunking, "mixed", mixed_result)
