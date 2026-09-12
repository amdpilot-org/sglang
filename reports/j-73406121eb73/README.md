# Independent review of PR 2365

Reviewed https://github.com/amdpilot-org/sglang/pull/2365 at exact commit
`838047177f7de105b83d6cd5ac7580c906720cee` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/33187
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/2398

Recommendation: **accept**. The candidate fully resolves the original issue at
source level in the paths that could be exercised here.

## Independent findings

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the proposed
`SGLANG_ABORT_ON_NAN_LOGITS` option does not exist. With existing sanitization
enabled, a full FP16 GPU row of 163,840 NaNs became all `-inf`; softmax remained
non-finite and summed to NaN. This reproduces the unsafe original behavior on
the available GPU, though not the reporter's Blackwell/Kimi-K3 deployment.

At the exact candidate commit, imports resolved to the checked-out source. The
candidate detects full-NaN rows before sanitization, makes abort-only mode safe
for sampling, transfers one mask bit per request with existing results, aborts
with HTTP 503 before committing a sampled token, and releases KV with
`is_insert=False`. Speculative verification rows are reduced to one bit per
request rather than reaching the prior assertion.

Independent GPU cases passed for FP16, BF16, and FP32, a 163,840-wide full-NaN
row paired with a partially-NaN healthy row, temperature 0.01 after
sanitization, opt-out behavior, and speculative masks with failures in multiple
requests. Candidate-focused tests passed (18 tests and 2 subtests).

No native source changed, and `repository-environment.json` declares no native
component, so a native rebuild was not applicable. Python compilation passed.
`git diff --check` reports trailing whitespace only inside the candidate's
committed archived pytest logs; this is a report-artifact hygiene issue, not a
functional counterexample.

## Limitations

- Hardware was one AMD Instinct MI350X (`gfx950`) with ROCm 7.2, not the
  reporter's Blackwell-class SM10x TP=8 deployment.
- Kimi-K3 weights, the transient production NaN source, multi-node execution,
  and a full HTTP model-serving reproduction were unavailable.
- No configured live hierarchical-cache storage backend was available. Cache
  non-publication was verified through the real `release_kv_cache(...,
  is_insert=False)` source path and candidate cleanup tests, but not an
  end-to-end host-cache write-through deployment.
