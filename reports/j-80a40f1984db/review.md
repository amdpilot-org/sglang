# Consolidated correction for NIXL clear scoping

Candidate: https://github.com/amdpilot-org/sglang/pull/2036 at
`aed3cae082a1786ded12e8d4e7d1f35c93e52807`

Independent review: https://github.com/amdpilot-org/sglang/pull/2127

The candidate's scoped clear, unusable-suffix refusal, and component handling are
retained. Its matcher searched for the requested suffix anywhere in a filename,
so `_model_0_1` matched the tail of `_tenant_model_0_1`; the same defect occurred
for MLA suffixes `_model` and `_tenant_model`.

Both cases were independently reproduced at the exact candidate commit. The
matcher is now anchored to the filename grammar emitted by the production path:
a 64-character lowercase SHA-256 page key, then the exact instance suffix, then
either end-of-name or an underscore-prefixed component. This also prevents an
unrelated filename containing the suffix from being removed.

Raw failing-before and passing-after logs are retained in `raw/`.
