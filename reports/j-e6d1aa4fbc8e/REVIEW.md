# Independent review of amdpilot-org/sglang PR 1075

Candidate: https://github.com/amdpilot-org/sglang/pull/1075 at `bd95273ff6988e1191c9d387783fd8ac15f8bcdb`

Upstream issue: https://github.com/sgl-project/sglang/issues/37128

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1112

## Recommendation

Accept. The exact candidate fully resolves the source-level contract documented by the open issue and its sole follow-up comment. No remaining counterexample was found.

## Review findings

There are no blocking findings.

The recorded base reproduces the defect: `set_spec_verify_start_time` and `set_spec_verify_end_time` have no production callers, EAGLE V2 has no draft timing calls, and `SPEC_DRAFT_EXTEND` is absent. The candidate adds draft start/end and verify start calls to both active EAGLE V2 implementations, adds verify start/end to NGRAM, and closes both EAGLE verify spans in their shared `run_eagle_verify` implementation.

The shared verify-end helper converts `accept_lens`, whose values include the target bonus token, to drafts-only counts before calling the existing setter. That setter emits both `num_correct_drafts` and the backward-compatible `accepted_tokens` attribute. The helper returns before touching the tensor when tracing is globally disabled. The candidate also correctly avoids opening or closing a draft span on the EAGLE zero-step path, where no draft operation runs.

The issue discussion explicitly treats `spec_draft_extend` as deliberately removed rather than part of the restoration contract. The candidate appropriately does not reintroduce it.

## Evidence and limitations

The candidate's 11 focused tests passed. An independent test used a real ROCm tensor on the assigned AMD Instinct MI350X (gfx950) and verified the zero-draft boundary plus drafts-only conversions `accept_lens [1, 2, 5] -> [0, 1, 4]`. It also verified disabled tracing avoids request setter calls and that empty batches avoid tensor operations. Python compilation and `git diff --check` passed.

No native files changed, and the prepared environment declares no native component, so a native rebuild was not applicable. No EAGLE or NGRAM weights were available; consequently this review does not claim a live model-serving/exporter trace, semantic model validation, multi-GPU validation, or multi-node validation. The issue itself defines a deterministic source/call-site reproduction, and the candidate plus focused behavioral tests directly satisfy that contract.

Raw evidence is retained under `reports/j-e6d1aa4fbc8e/raw/`; the complete checkout-independent review workspace remains at `/job/review-evidence-j-e6d1aa4fbc8e/` for the job lifetime.
