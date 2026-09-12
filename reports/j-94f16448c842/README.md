# MiniMax-M3 thinking-control investigation

The recorded base silently discarded `thinking={"type":"disabled"}` from
`ChatCompletionRequest`. The raw before probe shows that the official field
produced neither a retained request attribute nor the existing
`chat_template_kwargs.thinking_mode` control. The escape hatch supplied by the
report was retained correctly.

The fix declares the two supported MiniMax values and maps them to the existing
template contract:

- `disabled` -> `thinking_mode="disabled"`
- `adaptive` -> `thinking_mode="enabled"`

An explicitly supplied `thinking_mode` wins, so the compatibility mapping does
not override lower-level caller intent. Unsupported values now fail request
validation rather than being silently ignored.

Evidence is retained under `raw/`, including the source issue's current open
candidate PR diff, the direct before/after normalization probes, and the
failing-before/passing-after test output.

This validates the actual serving request normalization and template-control
handoff. It does not validate MiniMax-M3 generation semantics: the model weights
and the reported TP=4 topology were unavailable, and the assigned single GPU
cannot reproduce that distributed launch.
