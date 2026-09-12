from sglang.srt.parser.reasoning_parser import ReasoningParser


def stream(chunks, *, force_reasoning=False):
    parser = ReasoningParser(model_type="qwen3", force_reasoning=force_reasoning)
    reasoning = content = ""
    for chunk in chunks:
        next_reasoning, next_content = parser.parse_stream_chunk(chunk)
        reasoning += next_reasoning or ""
        content += next_content or ""
    next_reasoning, next_content = parser.parse_stream_end()
    return reasoning + (next_reasoning or ""), content + (next_content or "")


text = "Sure.<think>checking</think>Done."
expected = ("checking", "Sure.Done.")

assert ReasoningParser(model_type="qwen3").parse_non_stream(text) == expected
assert stream(["Sure.", "<think>", "checking", "</think>", "Done."]) == expected
assert stream(["Sure.<", "think>", "checking", "</think>", "Done."]) == expected
assert stream([text]) == expected
assert ReasoningParser(model_type="qwen3").parse_non_stream(
    "\n<think>checking</think>Done."
) == ("checking", "\nDone.")
assert stream(["answer <"]) == ("", "answer <")
assert stream(["lead<think>r</think>tail"], force_reasoning=True) == (
    "leadr",
    "tail",
)

print("reasoning parser regression passed")
