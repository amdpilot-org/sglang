import json
import logging

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.trinity_detector import TrinityDetector

TOOLS = [Tool(function=Function(name=n, parameters={})) for n in ("get_weather", "get_time")]
W = '{"name":"get_weather","arguments":{"city":"Tokyo"}}'
T = '{"name":"get_time","arguments":{"tz":"JST"}}'
X = '{"name":"rm_rf","arguments":{"path":"/"}}'

def collect(detector, wire, size, finish=True):
    chunks = list(wire) if size == 1 else ([wire] if size == 0 else [wire[i:i+size] for i in range(0, len(wire), size)])
    calls, normal = {}, ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, TOOLS)
        normal += result.normal_text or ""
        for call in result.calls or []:
            slot = calls.setdefault(call.tool_index, {"name": None, "arguments": ""})
            slot["name"] = call.name or slot["name"]
            slot["arguments"] += call.parameters or ""
    if finish:
        result = detector.finish(TOOLS)
        normal += result.normal_text or ""
        for call in result.calls or []:
            slot = calls.setdefault(call.tool_index, {"name": None, "arguments": ""})
            slot["name"] = call.name or slot["name"]
            slot["arguments"] += call.parameters or ""
    return calls, normal

def wires(seq):
    return {
        "json": (JsonArrayParser, "[" + ",".join(x.replace('arguments', 'parameters') for x in seq) + "]"),
        "qwen": (Qwen25Detector, "".join(f"<tool_call>\n{x}\n</tool_call>\n" for x in seq)),
        "hermes": (HermesDetector, "".join(f"<tool_call>{x}</tool_call>" for x in seq)),
        "llama": (Llama32Detector, "<|python_tag|>" + ";".join(seq)),
        "mistral": (MistralDetector, "[TOOL_CALLS] [" + ",\n\t".join(seq) + "]"),
        "trinity": (TrinityDetector, "".join(f"<tool_call>\n{x}\n</tool_call>\n" for x in seq)),
    }

expected = {0: {"name":"get_weather", "arguments":'{"city":"Tokyo"}'}, 1: {"name":"get_time", "arguments":'{"tz":"JST"}'}}
def canonical(calls):
    return {index: {"name": value["name"], "arguments": json.loads(value["arguments"])} for index, value in calls.items()}
expected_canonical = {0: {"name":"get_weather", "arguments":{"city":"Tokyo"}}, 1: {"name":"get_time", "arguments":{"tz":"JST"}}}
failed = []
for order, seq in {"first":(X,W,T), "middle":(W,X,T), "last":(W,T,X), "double":(W,X,X,T)}.items():
    for fmt, (cls, wire) in wires(seq).items():
        for size in (1, 0, 7, 31):
            got, normal = collect(cls(), wire, size)
            try:
                ok = canonical(got) == expected_canonical and normal.strip() == ""
            except (json.JSONDecodeError, TypeError):
                ok = False
            print(json.dumps({"order":order,"format":fmt,"chunk":size,"ok":ok,"calls":got,"normal":normal}, sort_keys=True))
            if not ok: failed.append((order,fmt,size))
for order, seq in {"first":(X,W,T), "middle":(W,X,T)}.items():
    for fmt, (cls, wire) in wires(seq).items():
        got, normal = collect(cls(), wire, 1, finish=False)
        try:
            ok = canonical(got) == expected_canonical and normal.strip() == ""
        except (json.JSONDecodeError, TypeError):
            ok = False
        print(json.dumps({"order":order,"format":fmt,"chunk":1,"finish":False,"ok":ok,"calls":got,"normal":normal}, sort_keys=True))
        if not ok: failed.append((order,fmt,"no-finish"))
print(f"SUMMARY failures={len(failed)} cases={4*6*4+2*6}")
raise SystemExit(bool(failed))
