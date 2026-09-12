# Independent review of amdpilot-org/sglang PR 2958

Reviewed exact commit `5f47ca6feb445c0a130ed1231f06ddccbb0588ee` against upstream issue https://github.com/sgl-project/sglang/issues/32264 and the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **unverified**. The candidate corrects the three concrete admission defects reported against `333b37951dfca5d98dffb77b25ee5a8e0801100b`, but this review cannot qualify the full original Apple Silicon/MLX contract.

## Findings

- On the recorded base, the MLX Gemma 4 MTP modules and candidate regression do not exist; collecting the extracted regression fails.
- On prior candidate `333b379…`, independent requests using `temperature=0.7, top_k=1` and `temperature=0, beam_width=2` both pass admission.
- On reviewed candidate `5f47ca6…`, both are rejected. Explicit zero survives Python normalization and a msgpack encode/decode round-trip, while default/nonzero temperature does not gain the zero marker.
- Shape-compatible alternative target/assistant repositories and mutable revisions are rejected; the exact Level 1 repositories and revisions are required.
- The focused guardrail suite passes (5 tests, 27 subtests). The full MLX collection reports 29 passed, 214 skipped, and 42 subtests passed.
- The only Stage B E2E is skipped because Apple Silicon, MLX, and mlx-vlm are unavailable.
- Python imports resolve to `/job/repo/python`. `mlx` does not resolve. No C++ code changed. Rust sampling transport changed, but `cargo` is unavailable, so it was not rebuilt.

## Scope conclusion

This is a verified correction of the known admission counterexamples, not verification of the complete original issue. Real MLX numerical parity, native cache transactions, counters, cleanup, flush, and post-flush serving remain unexecuted. The tiny Llama transport fixture is intentionally not used as proof for a Gemma 4/MLX architectural and numerical contract.

Raw evidence was retained outside the checkout under `/job/review_evidence/` while revisions were switched.
