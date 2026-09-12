"""Independent review case for PR 1754 at 2ce6f8e16316f58a3f88569030a95778238a0fec."""

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mistral_detector import MistralDetector

TOOLS = [
    Tool(function=Function(name="get_weather", parameters={})),
    Tool(function=Function(name="get_time", parameters={})),
]
WEATHER = '{"name":"get_weather","arguments":{"city":"Tokyo"}}'
UNKNOWN = '{"name":"rm_rf","arguments":{"path":"/"}}'
TIME = '{"name":"get_time","arguments":{"tz":"JST"}}'
EXPECTED = {
    0: {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
    1: {"name": "get_time", "arguments": '{"tz": "JST"}'},
}


def collect(separator):
    detector = MistralDetector()
    calls = {}
    normal_text = ""
    wire = "[TOOL_CALLS] [" + separator.join((WEATHER, UNKNOWN, TIME)) + "]"
    for chunk in wire:
        result = detector.parse_streaming_increment(chunk, TOOLS)
        normal_text += result.normal_text or ""
        for call in result.calls or []:
            slot = calls.setdefault(
                call.tool_index, {"name": None, "arguments": ""}
            )
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


failures = []
for separator in (", ", ",", ",\n", ",\t"):
    observed = collect(separator)
    print(repr(separator), observed)
    if observed != (EXPECTED, "", ""):
        failures.append(separator)

assert not failures, f"canonical JSON separators still failing: {failures!r}"
