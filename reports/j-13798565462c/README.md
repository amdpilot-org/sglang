# Request logger list truncation investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/33164

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1945

At the recorded base, the JSON elision branch copied raw head and tail elements,
and the text formatter stringified list and tuple elements without recursion. The
reduced reproduction in `raw/reproduction-before.txt` demonstrates both defects:
with `max_length=8`, an eight-element list truncated each 10,000-character string
to 11 characters, while a nine-element list retained 10,000-character strings;
the text formatter also retained complete strings.

The implementation now recursively formats every retained list or tuple element.
`raw/reproduction-after.txt` records the original million-character fixture after
the correction, and `raw/focused-tests-after.txt` records the focused regression
suite. No GPU or native rebuild was used because both affected functions are pure
Python serialization helpers.
