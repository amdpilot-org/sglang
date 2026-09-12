from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector

tools = [
    Tool(
        type="function",
        function=Function(
            name=name,
            description=name,
            parameters={"type": "object", "properties": {}},
        ),
    )
    for name in ("first", "second")
]
detector = Glm47MoeDetector()
text = "<tool_call>first</tool_call><tool_call>second</tool_call>"
result = detector.parse_streaming_increment(text, tools)
print([(call.tool_index, call.name, call.parameters) for call in result.calls])
print(repr(detector._buffer))
