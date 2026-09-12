# Investigation summary

The checked-out `main` at `358c163250ad3b1f62939b01ce1314a0a31a0365`
still sanitized a fully-NaN logits row to a constant without retaining a failure
signal. On the assigned gfx950, a 163,840-wide row became exactly uniform after
softmax (`max_abs_error=0.0`), reproducing the issue's sampling mechanism.

The change adds opt-in `SGLANG_ABORT_ON_NAN_LOGITS` handling for the reported
plain prefill/decode path. Detection occurs after custom logit processors and
before sanitization. The per-request mask rides the existing overlap result copy.
Flagged requests receive a retriable 503 before the random token is committed,
and their KV is released without radix-cache insertion. Partial rows continue to
use sanitization, and the disabled flag has no tensor-scan overhead.

Related upstream work was inspected before implementation:

- https://github.com/sgl-project/sglang/pull/33206 (open)
- https://github.com/sgl-project/sglang/pull/33263 (closed)

The implementation does not claim to find or repair the unavailable Kimi-K3
kernel/model source of transient NaNs. Speculative decoding is explicitly outside
the qualified scope because its output rows require algorithm-specific mapping.

Raw GPU, test, lint, issue, and related-PR evidence is retained under `raw/`.
