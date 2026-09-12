# GLM-V literal placeholder investigation

- Upstream issue: https://github.com/sgl-project/sglang/issues/37579
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/953
- Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared base did not contain a fix. Upstream PR
https://github.com/sgl-project/sglang/pull/37900 was open and unmerged when
reviewed. It proposed GLM-family message-boundary neutralization based on the
existing Kimi K3 handling.

`reproduce_processor_mismatch.py` exercises the checked-out
`BaseMultimodalProcessor.load_mm_data` implementation with GLM-V's real image
and video placeholder strings and a stub media decoder. One wrapped image
placeholder, one literal image placeholder in text, and one image data item
reproduce the reported warning and empty-suffix `RuntimeError`; see
`reproduction-before.log` (exit 1).

The correction neutralizes literal GLM-V image/video placeholder strings while
messages still distinguish text from structured media. It is gated by both the
GLM-V model architecture and the presence of multimodal input. Structured media
parts therefore remain available for the chat template to render as real
wrapped placeholders. The same protection covers assistant reasoning, tool-call
arguments, tool-result text, and tool definitions. Native `/generate` and the
generic multimodal processor remain unchanged.

Regression coverage verifies image and video literals with a real structured
image, reasoning/tool fields, tool descriptions, preservation of structured
media, request immutability, all registered GLM-V architectures, non-GLM
behavior, and the explicitly unaffected text-only GLM-V boundary.

The prepared host exposes one AMD Instinct MI350X (`gfx950`), recorded in
`gpu-inventory.log`. No GPU execution was needed or claimed because the defect
and correction are deterministic CPU-side prompt preparation. GLM-5.3-Flash
weights were unavailable, so a full HTTP/model reproduction and semantic model
accuracy are not verified.
