from copy import deepcopy

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat


IMAGE = "<|image|>"
VIDEO = "<|video|>"


def render(messages, tools):
    chunks = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    chunks.append(part["text"])
                elif part.get("type") == "image_url":
                    chunks.append(f"<|begin_of_image|>{IMAGE}<|end_of_image|>")
                elif part.get("type") == "video_url":
                    chunks.append(f"<|begin_of_video|>{VIDEO}<|end_of_video|>")
        if message.get("reasoning_content"):
            chunks.append(message["reasoning_content"])
        for call in message.get("tool_calls") or []:
            chunks.append(str(call["function"].get("arguments", "")))
    chunks.append(str(tools or ""))
    return "\n".join(chunks)


messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"discuss {IMAGE} and {VIDEO}"},
            {"type": "image_url", "image_url": {"url": "image"}},
            {"type": "video_url", "video_url": {"url": "video"}},
        ],
    },
    {
        "role": "assistant",
        "content": f"assistant {IMAGE}",
        "reasoning_content": f"reasoning {VIDEO}",
        "tool_calls": [
            {"function": {"name": "inspect", "arguments": {"literal": IMAGE}}}
        ],
    },
    {"role": "tool", "content": f"tool result {VIDEO}"},
]
tools = [{"function": {"name": "inspect", "description": f"docs {IMAGE}"}}]

before = render(deepcopy(messages), deepcopy(tools))
after = render(
    OpenAIServingChat._prepare_glm_v_messages(deepcopy(messages)),
    __import__(
        "sglang.srt.entrypoints.openai.serving_chat", fromlist=["x"]
    ).neutralize_glm_v_placeholder_value(deepcopy(tools)),
)

print("before_image_count", before.count(IMAGE))
print("before_video_count", before.count(VIDEO))
print("after_image_count", after.count(IMAGE))
print("after_video_count", after.count(VIDEO))
print("input_preserved", messages[0]["content"][0]["text"])
assert before.count(IMAGE) == 5
assert before.count(VIDEO) == 4
assert after.count(IMAGE) == 1
assert after.count(VIDEO) == 1
assert messages[0]["content"][0]["text"] == f"discuss {IMAGE} and {VIDEO}"
