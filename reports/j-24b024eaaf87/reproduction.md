# Qwen-VL marker correction generation 2

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Candidate reproduced: https://github.com/amdpilot-org/sglang/pull/3435 at `1fefcc22ba785ac966c77bef6c7a9a93a4b3fae4`

Independent review investigated: https://github.com/amdpilot-org/sglang/pull/3439

The candidate's valid one-image/one-authoritative-marker behavior was retained. Added adversarial assertions independently reproduced all three reviewed failures: mixed user and tool literals were encoded only after replacement with spaced spellings, and a complete marker divided across adjacent text parts bypassed per-part rewriting, leaving two loader markers for one real PNG and triggering the legacy loader's empty-detail `RuntimeError` path.

The correction keeps the exact client-faithful render and its encoding, builds a distinct loader-safe render only for mixed Qwen-VL image requests, and treats adjacent same-kind text parts as a continuous stream before neutralization. The loader therefore observes one authoritative marker for one image even when an untrusted marker crosses a part boundary.

Raw failing-before and passing-after output is under `evidence/`. Qwen3.8-27B weights were unavailable, so this qualifies the checked-out prompt-rendering and actual multimodal-loading boundary with a real PNG, not full model serving or semantic inference.
