import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.trinity_detector import TrinityDetector

TOOLS = [
    Tool(function=Function(name="get_weather", parameters={})),
    Tool(function=Function(name="get_time", parameters={})),
]
W = '{"name":"get_weather","arguments":{"city":"Tokyo"}}'
X = '{"name":"rm_rf","arguments":{"path":"/"}}'
T = '{"name":"get_time","arguments":{"tz":"JST"}}'


def collect(detector, chunks, finish=True):
    calls, text = {}, ""
    results = [detector.parse_streaming_increment(c, TOOLS) for c in chunks]
    if finish:
        results.append(detector.finish(TOOLS))
    for result in results:
        text += result.normal_text or ""
        for call in result.calls or []:
            slot = calls.setdefault(call.tool_index, {"name": None, "arguments": ""})
            if call.name:
                slot["name"] = call.name
            if call.parameters:
                slot["arguments"] += call.parameters
    return calls, text, detector._buffer


def wires(detector):
    if isinstance(detector, JsonArrayParser):
        conv = lambda value: value.replace('"arguments"', '"parameters"')
        prefix, separator, suffix = "[", ",", "]"
        return prefix + conv(W), separator + conv(X) + separator + conv(T) + suffix
    if isinstance(detector, MistralDetector):
        return "[TOOL_CALLS] [" + W, ", " + X + ", " + T + "]"
    if isinstance(detector, Llama32Detector):
        return "<|python_tag|>" + W, ";" + X + ";" + T
    if isinstance(detector, HermesDetector):
        return (
            "<tool_call>" + W,
            "</tool_call><tool_call>"
            + X
            + "</tool_call><tool_call>"
            + T
            + "</tool_call>",
        )
    return (
        "<tool_call>\n" + W,
        "\n</tool_call>\n<tool_call>\n"
        + X
        + "\n</tool_call>\n<tool_call>\n"
        + T
        + "\n</tool_call>",
    )


def main():
    mistral_wire = "[TOOL_CALLS] [" + ", ".join((W, X, T)) + "]"
    print("mistral_char", collect(MistralDetector(), mistral_wire, finish=False))
    for cls in (
        JsonArrayParser,
        Qwen25Detector,
        HermesDetector,
        Llama32Detector,
        TrinityDetector,
        MistralDetector,
    ):
        detector = cls()
        first, final = wires(detector)
        print(cls.__name__, collect(detector, [*first, final]))
    wire = (
        "["
        + ",".join(v.replace('"arguments"', '"parameters"') for v in (W, X, T))
        + "]"
    )
    chunks = [wire[i : i + 31] for i in range(0, len(wire), 31)]
    print("coarse31", collect(JsonArrayParser(), chunks))


if __name__ == "__main__":
    main()
