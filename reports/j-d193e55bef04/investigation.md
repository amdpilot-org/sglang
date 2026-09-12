# Investigation notes

- Upstream issue: https://github.com/sgl-project/sglang/issues/37326
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/986
- Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Current `main` still contains the two speculative tracking expressions and the
legacy `MambaRadixCache` condition described in the report. Those are not
changed here because later evidence on the upstream issue says a deployment
with the legacy fix and both clamps still relapsed after about 16 hours.

The active registry constructs `UnifiedRadixCache`, including for the current
Qwen4Exp hybrid cache configuration. Its unfinished-request path already skips
insertion when `effective_cache_len <= 0`; the finished-request path did not.
The GPU fixture reproduced the resulting ownership error deterministically:
an empty insert returned the root node, causing a finished request that held a
different matched-node lock to be repointed to the unlocked root. The patch
keeps the previous node handle, releases the held lock, returns all request
KV/Mamba slots, and prevents a zero-length session reference.

Related change inspected before implementation:
https://github.com/sgl-project/sglang/pull/38191 at commit
`68957889039027f5003bddf469b866d45323459a`. The narrow behavior is adapted to
the newer prepared base rather than importing that older PR's unrelated diff.

The original long-uptime model-level symptom remains unverified because this
job has one AMD gfx950 GPU and no reported 176B checkpoint weights, while the
report used two SM121 nodes. The result is therefore `candidate_verified`, not
`fixed` or `reproduced`.
