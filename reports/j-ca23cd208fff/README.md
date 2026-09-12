# Issue 38013 investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/38013

Mirror issue: https://github.com/amdpilot-org/sglang/issues/822

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the actual
`DeepSeekV4Detector` forwarded all four structurally valid malformed shapes from
the report unchanged: an `arguments` object, JSON-string `arguments`, `args`,
and repeated `arguments` wrapping. The raw before/after outputs are retained in
`raw/reproduction-before.txt` and `raw/reproduction-after.txt`.

The correction is limited to DeepSeek V4. Once a DSML invoke is complete, it
uses that function's declared top-level JSON-schema properties to distinguish a
model-added transport envelope from a real parameter. It peels `arguments` or
`args` only when the envelope name is not itself declared and the eventual
object keys are declared tool properties. V4 direct-JSON arguments are buffered
until invoke completion so streaming clients never receive a prefix that later
needs rewriting. XML parameter-tag calls retain incremental streaming.

Boundary tests demonstrate that a legitimate parameter named `arguments`, an
unknown inner object, a corrupted JSON string, and an already-correct payload
are not rewritten. Corrupted command text cannot be reconstructed safely.

The assigned MI355X/gfx950 was visible, as recorded in `raw/gpu-inventory.txt`,
but no GPU execution was relevant to this parser-only correction. The named
DeepSeek-V4-Flash-Vision-Exp weights were unavailable. Consequently this report
does not claim a full-model, vision-semantic, retry-loop, or distributed
reproduction. The deterministic tiny Llama fixture would only test transport
and a different engine architecture, so it was not presented as evidence for
this model-specific output behavior.
