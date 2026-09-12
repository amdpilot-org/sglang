from sglang.srt.parser.reasoning_parser import InklingDetector


JSON = "<|content_invoke_tool_json|>"
TEXT = "<|content_invoke_tool_text|>"
END = "<|end_message|>"
MODEL = "<|message_model|>"


def streamed(source, sizes, *, continue_final_message=False):
    detector = InklingDetector(continue_final_message=continue_final_message)
    output = []
    pos = 0
    index = 0
    while pos < len(source):
        size = sizes[index % len(sizes)]
        output.append(detector.parse_streaming_increment(source[pos : pos + size]).normal_text)
        pos += size
        index += 1
    output.append(detector.finish().normal_text)
    return "".join(output)


cases = {
    "json header": (f"weather{JSON}{{\"name\":\"weather\",\"args\":{{}}}}{END}", f"{MODEL}weather{JSON}{{\"name\":\"weather\",\"args\":{{}}}}{END}"),
    "text header": (f"search{TEXT}query{END}", f"{MODEL}search{TEXT}query{END}"),
    "empty header": (f"{JSON}{{\"name\":\"weather\",\"args\":{{}}}}{END}", f"{MODEL}{JSON}{{\"name\":\"weather\",\"args\":{{}}}}{END}"),
    "plain text": ("ordinary answer", "ordinary answer"),
    "text then model block": (f"prefix{MODEL}<|content_text|>suffix{END}", f"prefixsuffix"),
}

for name, (source, expected) in cases.items():
    one_shot = InklingDetector().detect_and_parse(source).normal_text
    assert one_shot == expected, (name, "one-shot", one_shot, expected)
    for sizes in ([1], [2, 7, 3], [len(source)]):
        actual = streamed(source, sizes)
        assert actual == expected, (name, sizes, actual, expected)

detector = InklingDetector(continue_final_message=True)
increment = detector.parse_streaming_increment(" resumed answer").normal_text
assert increment == " resumed answer", increment
assert detector.finish().normal_text == ""
print(f"PASS: {len(cases)} one-shot cases, {len(cases) * 3} chunked cases, and continue_final_message latency")
