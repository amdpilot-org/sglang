import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector


def wrapped(payload: str) -> str:
    return (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="run">'
        + payload
        + '</｜DSML｜invoke></｜DSML｜tool_calls>'
    )


def tool(parameters):
    return Tool(type="function", function=Function(name="run", parameters=parameters))


def parse(payload, parameters):
    result = DeepSeekV4Detector().detect_and_parse(
        wrapped(json.dumps(payload)), [tool(parameters)]
    )
    return result.calls[0].parameters


def stream(payload, parameters):
    text = wrapped(json.dumps(payload))
    detector = DeepSeekV4Detector()
    pieces = []
    for index in range(0, len(text), 3):
        result = detector.parse_streaming_increment(
            text[index : index + 3], [tool(parameters)]
        )
        pieces.extend(call.parameters for call in result.calls)
    return "".join(pieces)


schema = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "file_path": {"type": "string"},
    },
}
cases = {
    "clean_quadruple": {
        "arguments": {
            "arguments": {
                "arguments": {"arguments": {"command": "printf ok"}}
            }
        }
    },
    "duplicated_inner_command": {
        "arguments": {"command": "printf okprintf ok"}
    },
    "damaged_inner_json_string": {"arguments": '{"command":"printf ok"'},
    "reported_corrupt_shape_as_string": {
        "arguments": {
            "arguments": {
                "arguments": {
                    "arguments": '{"file_path":"path": "x"}'
                }
            }
        }
    },
}

for name, payload in cases.items():
    print(name)
    print("  one_shot:", parse(payload, schema))
    print("  streaming:", stream(payload, schema))

for name, parameters in {
    "empty_schema": {},
    "object_without_properties": {"type": "object"},
    "freeform_object": {"type": "object", "additionalProperties": True},
}.items():
    payload = {"arguments": {"command": "printf ok"}}
    print(name)
    print("  one_shot:", parse(payload, parameters))
    print("  streaming:", stream(payload, parameters))
