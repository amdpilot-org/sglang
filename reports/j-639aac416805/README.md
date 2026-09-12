# Issue 34366 investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/34366

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1616

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`,
`ImageEncodingStage.forward` forwarded `image_grid_thw` but reconstructed the
text-encoder arguments without the processor's `mm_token_type_ids`. The prepared
environment uses Transformers 5.12.1, whose Qwen3-VL implementation explicitly
raises when multimodal grids and input IDs are provided without that field.

The correction conditionally includes `mm_token_type_ids` in the shared argument
builder used by both the positive and classifier-free-guidance calls. Conditional
inclusion is intentional so older/non-Qwen processors without the field retain
their existing call shape.

The full JoyAI model was not downloaded or served. The regression isolates the
reported processor-to-encoder boundary, and the GPU fixture only establishes
that the device tensor is passed through unchanged on the assigned gfx950 GPU.
See `result.json` and `raw/` for commands, outcomes, and retained output.
