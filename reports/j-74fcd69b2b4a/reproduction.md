# Qwen-VL literal marker reproduction

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The failing-before test drove `OpenAIServingChat._apply_jinja_template()` with a
Qwen-VL-style template, one structured `image_url`, and either user text or tool
result text containing `<|vision_start|><|image_pad|><|vision_end|>`. The actual
rendered prompt contained two complete markers while `image_data` contained one
item. This is the condition that routes `BaseMultimodalProcessor.load_mm_data()`
to `legacy_load_mm_data()`, where the second regex match exhausts the image
iterator and becomes the reported empty-detail runtime error.

Before the implementation change, the focused test failed for both roles with:

```text
AssertionError: 2 != 1
```

After the change, the template-emitted attachment marker remains exact and the
client-supplied marker is rendered as readable ordinary text:

```text
<| vision_start |><| image_pad |><| vision_end |>
```

The resulting prompt contains one parser-recognized marker for one image. The
test then passes the rendered prompt and a real 1x1 PNG data URL to the actual
multimodal loader and verifies that exactly one PIL image is decoded.

Upstream PR https://github.com/sgl-project/sglang/pull/35585 was reviewed before
delivery. It changes the exhausted iterator into a `ValueError`/HTTP 400 and
explicitly leaves literal reserved-token handling unresolved. This patch keeps
malformed attachment references as client validation failures while preserving
valid ordinary text as non-attachment content.
