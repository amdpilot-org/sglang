# Investigation of qwen3.6-35b-a3b repeated output

Upstream issue: https://github.com/sgl-project/sglang/issues/36276

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1214

## Outcome

The exact report could not be reproduced in the prepared environment because the
reported `Qwen3.6-35B-A3B-FP8` checkpoint (including its EAGLE draft weights and
model revision) is not available. The issue also omits the request body and raw
server logs. No model or parser code was changed without that evidence.

The attached video was downloaded to
`/tmp/amdpilot-repo-j-4dd16c1f5ba1/issue-36276.mp4` and decoded successfully. It
shows a normal assistant explanation degenerating into repeated text. It does
not show duplicate HTTP events or repeated structured tool-call objects, so the
video does not support attributing the symptom to `qwen3_coder` parsing.

## Related fixes already in the checkout

Current source already marks the final linear-attention and full-attention
Qwen3.5-family decoder layers with `is_last_layer`. This is the fix merged in
sgl-project/sglang#19411 for a previously demonstrated repetitive-output bug
caused by deferring the final tensor-parallel all-reduce.

The current implementation has a second boundary safeguard in both decoder
paths: if all-reduce fusion is selected on the final layer without the supported
deferred MoE-finalize handoff, it disables deferral so `postprocess_layer`
performs the collective. That safeguard entered this source after the reported
issue as part of sgl-project/sglang#35758.

These changes are plausible protections against the class of numerical
corruption visible in the report, but the report does not provide enough data
to prove that either was its cause. In particular, the earlier
`is_last_layer` fix predates this report, while the later fallback is associated
with a different fused collective path. They are therefore recorded as related
evidence, not claimed as a verified resolution.

## Validation

- `python -m pytest -q test/registered/unit/function_call/test_function_call_parser.py -k 'qwen3_coder_detect_and_parse or qwen3_coder_streaming'`
  passed two independent boundaries: one-shot and chunked streaming parsing.
- `python -m pytest -q test/registered/unit/layers/moe/test_qwen35_flashinfer_fusion.py`
  passed all ten focused fusion tests.
- A source-history regression checked `5f216fc3^`, `5f216fc3`, and the current
  checkout. The final-layer fallback is absent before the change and present in
  both the linear-attention and full-attention decoder paths afterward and now.
- The assigned device is visible as one AMD Instinct MI350X,
  `gfx950:sramecc+:xnack-`, under Torch `2.11.0+rocm7.2` / HIP `7.2.26015`.
  This was inventory only; no issue-equivalent model execution occurred, so
  `gpu_execution` is reported as false.

Raw command output is retained in `evidence/`. The decoded issue video and
sample frames remain in the private runtime directory outside the checkout.

## Remaining limitations

- No `Qwen3.6-35B-A3B-FP8` target or EAGLE draft weights were available.
- The original request payload, tool schema, response stream, logs, exact image
  digest, tensor-parallel size, GPU type, and model revision were not supplied.
- A tiny Llama fixture cannot exercise the hybrid Qwen3.6 architecture, FP8 KV
  cache, Mamba `extra_buffer`, Qwen MTP/EAGLE draft, or model semantic output, so
  it was intentionally not substituted for the reported reproduction.
- The available single gfx950 GPU cannot reproduce a multi-GPU collective bug
  if the reporter's unspecified deployment used tensor parallelism.
