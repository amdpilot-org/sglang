# Independent review of PR 1053

Upstream issue: https://github.com/sgl-project/sglang/issues/37216

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1088

Candidate: https://github.com/amdpilot-org/sglang/pull/1053 at `6a5b8dc9c3b48604843c02b8e6bca139464b2c59`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept. The candidate fixes the issue-specific metadata identity consistently across all three relevant points: seed registration, bootstrap storage, and destination lookup. SGLang constructs the distributed rank as `tp_size * pp_rank + tp_rank` in the ordinary model-parallel case, so TP1/PP2 stages use keys 0 and 1 instead of both overwriting TP key 0.

This is a source-level full fix for the reported collision. It is not merely test hardening: the recorded base fails the candidate regression in the expected producer/server/consumer locations, while the exact candidate passes. Independent live-HTTP tests also preserved two PP-stage entries, exercised world rank 9 for a nontrivial TP4/PP layout, and checked missing, negative, malformed, legacy, overwrite, and isolation behavior.

## Evidence

- `raw/base-regression.log`: exact recorded base; 5 failed, 4 passed, and 14 subtests passed. Failures demonstrate the old `tp_rank` protocol and absent world-rank plumbing.
- `raw/candidate-tests.log`: exact candidate; 9 passed and 14 subtests passed. It also records that `sglang`, the bootstrap server, and loader imported from `/job/repo/python`.
- `raw/independent.log`: independent live HTTP protocol cases, syntax checks, architecture inventory, and confirmation that no native source file changed.
- `candidate.diff`: reviewed source and regression diff preserved outside revision switching.

## Limitations

The assigned host exposed one AMD Instinct MI355X with ROCm 7.2, not the report's two NVIDIA A100 GPUs. The Qwen3-8B weights and a second GPU were unavailable. Therefore this review did not run the full TP1/PP2 seed/client model launch, RDMA data transfer, multi-node behavior, model semantics, or the original CUDA architecture. The tested behavior is the actual HTTP bootstrap implementation plus the source producer/consumer call paths. No native code changed, so no native rebuild was required or performed.
