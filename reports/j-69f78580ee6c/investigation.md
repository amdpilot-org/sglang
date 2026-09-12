# Consolidated correction for Qwen3 gateway reasoning parsing

Upstream issue: https://github.com/sgl-project/sglang/issues/35148

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1658

Candidate: https://github.com/amdpilot-org/sglang/pull/1508 at `ac23f039ebd7e7dcbc44a11226265c82b1e95f18`

Independent review: https://github.com/amdpilot-org/sglang/pull/1623

The candidate's two focused Qwen3 tests passed when rerun at its exact commit. An independent candidate-style test then reproduced the review's tag-free counterexample: enabling the parser without a prefilled think tag classified `plain answer` entirely as reasoning. Source tracing also confirmed that the issue's `http://worker:8000` URL selects `ConnectionMode::Http`, whose passthrough router was untouched by the candidate.

This correction preserves the candidate's reasoning-parser upgrade and gRPC parsing support, requires both enabled thinking and `think_in_prefill` before initializing a parser inside a stripped prefilled think block, and applies the same non-streaming response split to the HTTP worker route. Existing non-null worker-provided `reasoning_content` is preserved.

No Qwen/Qwen3.8-27B-FP8 weights or NVIDIA L40S were available. The deterministic tests validate CPU-side routing and response parsing, not model semantics, architecture behavior, or HTTP streaming transformation.
