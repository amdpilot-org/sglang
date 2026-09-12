# NaN-logits containment correction generation 2

Reviewed candidate https://github.com/amdpilot-org/sglang/pull/2167 at exact
commit `40a488defc20fc508eb9f5d4e099ae869d2cbd9f` and independent review
https://github.com/amdpilot-org/sglang/pull/2249 against base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

The review's two concrete behavioral counterexamples reproduced:

- On the assigned AMD Instinct MI355X, abort-only FP16 sanitization converted a
  163,840-wide all-NaN row to all `-inf`; softmax remained NaN.
- The candidate's request-marking helper asserted for a speculative batch.

The correction preserves the candidate's request abort and cleanup changes,
makes all-NaN rows a finite zero row until host-side abort processing, and
reduces speculative verification masks from `[batch * draft_width]` to one bit
per request. Focused tests pass, including speculative mask reduction and
request-scoped cleanup. GPU checks pass for FP16, BF16, and FP32.

The hierarchical-cache safety claim remains bounded: mocked cleanup verifies
that abort release uses `is_insert=False`, but no model weights or configured
live hierarchical-cache storage backend were available for an end-to-end host
publication test. The gated model used by the broader sampling-mask suite was
also unavailable (HTTP 401); that is recorded as a limitation, not evidence for
another source change.

Raw outputs are retained in `raw/`.
