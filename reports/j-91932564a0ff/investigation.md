# Independent review of amdpilot-org/sglang PR 3352

Reviewed exact candidate commit `fd2b3990a384d1f86bb49d4ec0a7f69cf07d0837`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue:
https://github.com/sgl-project/sglang/issues/37160

Candidate provenance issue: https://github.com/amdpilot-org/sglang/issues/3327

## Recommendation

Accept the candidate as a narrow SGLang-side partial fix and fail-closed
mitigation. It does not fully resolve the original token-plus-salt external-key
contract until the companion LMCache API is available and deployed.

## Findings

- The recorded base reproduces the handoff defect in the actual implementation:
  `_mp_match_prefix` calls `lookup_kv(token_ids, req.rid)` while
  `key.cache_salt` is in scope, and `cache_finished_req` constructs
  `StoreMetadata` without `req.cache_salt`.
- The exact candidate forwards non-empty salts on MP lookup and store only when
  the connector advertises `supports_cache_salt is True`.
- With an older connector, salted requests skip external lookup and store. This
  prevents token-only cross-tenant external reuse, but it also means the
  requested salted external-cache behavior is unavailable.
- Unsalted requests retain the legacy two-argument lookup and unsalted store
  behavior.
- The candidate's `try/finally` makes MP store cleanup exception-safe.
- The candidate regression passed at the exact commit and failed on the base.
  The committed suite has five tests and all five failed on the base; the PR
  prose saying that four failed is a minor evidence-count discrepancy.
- Independent executable adversarial cases exercised the real candidate
  methods with controlled LMCache API stubs. They verified distinct salt
  forwarding, fail-closed behavior for absent/false/non-boolean capability
  flags, legacy unsalted compatibility, real `StoreMetadata.cache_salt`
  construction, and cleanup plus exception propagation after store failure.
- The candidate's store tests are primarily source assertions, so the
  independent executable store cases were necessary to validate behavior.

## Dependency and end-to-end limitation

The prepared interpreter has no `lmcache` package. Current LMCache `dev`,
inspected during this review, still defines `lookup_kv(token_ids, request_id)`
without a salt, defines `StoreMetadata` without `cache_salt`, and has no
`supports_cache_salt` capability. Companion LMCache PR 4842 is open and dirty;
its exact head does provide the API shape expected by this candidate. Therefore
the candidate cannot presently demonstrate token-plus-salt wire keys with a
released/current connector in this environment. A live LMCache daemon,
generated object keys, and the original latency/compute isolation scenario were
not reproduced.

## Architecture, GPU, native, and import evidence

- Prepared hardware: one AMD Instinct MI350X (`gfx950` class), ROCm 7.2,
  Torch `2.11.0+rocm7.2`.
- GPU execution was not used. The missing LMCache package/daemon is the relevant
  blocker, and an unrelated GPU numerical smoke cannot prove tenant-key
  isolation.
- No native source changed, so no native rebuild was required or performed.
- Candidate imports resolved SGLang from `/job/repo/python/sglang` and Torch from
  `/opt/venv/lib/python3.12/site-packages/torch`. Import of the LMCache-backed
  module through the normal path stopped with the expected missing-package
  `RuntimeError`; the independent boundary harness supplied controlled external
  API stubs without replacing SGLang source.

Raw revision-specific evidence was preserved outside the checkout under
`/job/review-evidence/` before returning to `amdpilot/j-91932564a0ff`.
