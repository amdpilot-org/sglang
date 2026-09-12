from sglang.srt.entrypoints.openai import encoding_dsv4


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the tool result."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"id":"demo"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "status=ok"},
    {
        "role": "system",
        "content": "<runtime_reminder>123 tokens left</runtime_reminder>",
    },
]

for final_role in ("system", "user"):
    case = [dict(message) for message in messages]
    case[-1]["role"] = final_role
    print(f"final_role={final_role}")
    try:
        rendered = encoding_dsv4.encode_messages(case, thinking_mode="thinking")
    except ValueError as error:
        print(f"error={error}")
        continue
    print(f"ends_with_generation_prefix={rendered.endswith('<｜Assistant｜><think>')}")
    print(repr(rendered[-200:]))
