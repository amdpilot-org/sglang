# Kimi-K2 streaming end-marker investigation

The recorded base already used marker-index streaming and handled multiple calls,
long nested arguments, prefix text, begin-marker fragments, and exception cleanup.
The remaining reproduced defect was asymmetric handling of marker boundaries:
`<|tool_call_begin|>` fragments were held, while a fragmented
`<|tool_call_end|>` prefix was immediately emitted as arguments.

The narrow source change holds only a trailing prefix of the end marker. On the
next increment it either becomes the complete marker or, if it diverges, is
released unchanged as argument text.

Raw logs in this directory preserve the failing-before and passing-after runs.
No native component was changed or rebuilt, and GPU execution is not relevant to
this CPU string parser.
